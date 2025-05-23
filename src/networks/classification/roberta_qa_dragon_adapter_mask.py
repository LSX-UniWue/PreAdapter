#coding: utf-8
import sys
import torch
from transformers import RobertaModel, RobertaConfig
import utils
from torch import nn
import torch.nn.functional as F
import random
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaForMaskedLM
from transformers import RobertaTokenizer

sys.path.append("./networks/base/")
from my_roberta_transformers import MyRobertaModel



class Net(torch.nn.Module):

    def __init__(self,taskcla, qa_task, args):

        super(Net,self).__init__()
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        args.build_adapter_mask = True
        self.roberta = MyRobertaModel.from_pretrained(args.bert_model,config=config,args=args)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)


        self.args = args
        self.qa_task = qa_task

        for param in self.roberta.parameters():
            if self.args.unfreez:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if args.apply_roberta_output and args.apply_roberta_attention_output:
            adapter_masks = \
                [self.roberta.encoder.layer[layer_id].attention.output.adapter_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.adapter_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_roberta_output:
            adapter_masks = \
                [self.roberta.encoder.layer[layer_id].output.adapter_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_roberta_attention_output:
            adapter_masks = \
                [self.roberta.encoder.layer[layer_id].attention.output.adapter_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        for adapter_mask in adapter_masks:
            for param in adapter_mask.parameters():
                param.requires_grad = True

        self.taskcla=taskcla


        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.gate=torch.nn.Sigmoid()
        self.config = config
        self.num_task = len(taskcla)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # init MLMHead, HF Work around
        original_model_for_mask_pred = RobertaForMaskedLM.from_pretrained(args.bert_model)
        # freeze and reuse classification layer
        kg_maskhead_layer = RobertaLMHead(config)
        kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.lm_head.state_dict())
        # freeze and reuse mask layer
        for param in kg_maskhead_layer.parameters():
            # param.requires_grad = True
            param.requires_grad = False

        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.roberta_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t_indx, _ in self.taskcla:
                if t_indx in self.qa_task.keys(): # qa task
                    self.last.append(torch.nn.Linear(args.roberta_hidden_size, 1))
                else: # kg task
                    self.last.append(kg_maskhead_layer)


        self.self_attns = nn.ModuleList()
        for t_num in range(args.ntasks):
            self.self_attns.append(Self_Attn(t_num+1))


        print(' ROBERTA KNOWLEDGE ADAPTER MASK')

        return

    def forward(self, t=None, input_ids=None, input_mask=None, start_mixup=None,s=None):
        output_dict = {}

        if start_mixup:
            sequence_output,pooled_output = \
                self.roberta(input_ids=input_ids, attention_mask=input_mask, t=t, s=s)
            masks = self.mask(t,s)
            pooled_output=self.self_attention_feature(t,input_ids, input_mask, pooled_output)

            pooled_output = self.dropout(pooled_output)

        else:

            sequence_output, pooled_output = \
                self.roberta(input_ids=input_ids, attention_mask=input_mask, t=t, s=s)
            masks = self.mask(t,s)
            pooled_output = self.dropout(pooled_output)

        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            for t_x, _ in self.taskcla:
                if t_x in self.qa_task.keys() and self.qa_task[t_x]['data_name'] in ['CSQA', 'OBQA']:
                    y.append(self.last[t_x](pooled_output))

                else:
                    y.append(self.last[t_x](sequence_output))

        output_dict['y'] = y
        output_dict['masks'] = masks
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)

        return output_dict



    def self_attention_feature(self,t,input_ids,segment_ids,input_mask,pooled_output):
        pre_pooled_outputs = []
        for pre_t in [x for x in range(t)]:
            with torch.no_grad():
                _,pre_pooled_output = \
                    self.roberta(input_ids=input_ids, attention_mask=input_mask, t=pre_t, s=self.args.smax)
            pre_pooled_outputs.append(pre_pooled_output.unsqueeze(-1).clone())


        pre_pooled_outputs = torch.cat(pre_pooled_outputs, -1)
        pre_pooled_outputs = torch.cat([pre_pooled_outputs,pooled_output.unsqueeze(-1).clone()], -1) # include itselves

        pooled_output = self.self_attns[t](pre_pooled_outputs) #softmax on task
        pooled_output = pooled_output.sum(-1) #softmax on task

        return pooled_output



    def mask(self,t,s):
        masks = {}
        for layer_id in range(self.config.num_hidden_layers):
            fc1_key = 'roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_mask.fc1' #gfc1
            fc2_key = 'roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_mask.fc2' #gfc2

            masks[fc1_key],masks[fc2_key] = self.roberta.encoder.layer[layer_id].attention.output.adapter_mask.mask(t,s)

            fc1_key = 'roberta.encoder.layer.'+str(layer_id)+'.output.adapter_mask.fc1' #gfc1
            fc2_key = 'roberta.encoder.layer.'+str(layer_id)+'.output.adapter_mask.fc2' #gfc2

            masks[fc1_key],masks[fc2_key] = self.roberta.encoder.layer[layer_id].output.adapter_mask.mask(t,s)


        return masks


    def last_mask(self,t,s):
        elast = self.elast(torch.LongTensor([t]).to(self.device))
        glast=self.gate(s*elast)
        return glast

    def get_view_for(self,n,p,masks):
        for layer_id in range(self.config.num_hidden_layers):
            if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_mask.fc1.weight':
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_mask.fc1.bias':
                return masks[n.replace('.bias','')].data.view(-1)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_mask.fc2.weight':
                post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
                return torch.min(post,pre)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_mask.fc2.bias':
                return masks[n.replace('.bias','')].data.view(-1)

            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_mask.fc1.weight':
                # print('not nont')
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_mask.fc1.bias':
                return masks[n.replace('.bias','')].data.view(-1)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_mask.fc2.weight':
                post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
                return torch.min(post,pre)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_mask.fc2.bias':
                return masks[n.replace('.bias','')].data.view(-1)


        return None


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,attn_size):
        super(Self_Attn,self).__init__()

        self.query_conv = nn.Linear(attn_size,attn_size)
        self.key_conv = nn.Linear(attn_size , attn_size)
        self.value_conv = nn.Linear(attn_size ,attn_size)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B,max_length,hidden_size)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        # print('x: ',x.size())
        m_batchsize,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,width,height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,width,height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        # print('energy: ',energy.size())

        attention = self.softmax(energy) # BX (N) X (N)

        # attention =  F.gumbel_softmax(energy,hard=True,dim=-1)
        # print('attention: ',attention)
        proj_value = self.value_conv(x).view(m_batchsize,width,height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,width,height)

        out = self.gamma*out + x


        return out
