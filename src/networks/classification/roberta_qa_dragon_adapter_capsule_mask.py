#coding: utf-8
import sys
import torch
from transformers import RobertaModel, RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaForMaskedLM
import utils
from torch import nn

sys.path.append("./networks/base/")
from my_roberta_transformers import MyRobertaModel
from transformers import RobertaTokenizer

class Net(torch.nn.Module):

    def __init__(self,taskcla, qa_task, args):

        super(Net,self).__init__()
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        args.build_adapter_capsule_mask = True
        self.roberta = MyRobertaModel.from_pretrained(args.bert_model,config=config,args=args)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)

        self.qa_task = qa_task

        for param in self.roberta.parameters():
            param.requires_grad = False


        if args.apply_roberta_output and args.apply_roberta_attention_output:
            adapter_masks = \
                [self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_roberta_output:
            adapter_masks = \
                [self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_roberta_attention_output:
            adapter_masks = \
                [self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        for adapter_mask in adapter_masks:
            for param in adapter_mask.parameters():
                param.requires_grad = True

        self.taskcla=taskcla

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.gate = torch.nn.Sigmoid()

        # init MLMHead, HF Work around
        original_model_for_mask_pred = RobertaForMaskedLM.from_pretrained(args.bert_model)
        # freeze and reuse classification layer
        kg_maskhead_layer = RobertaLMHead(config)
        kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.lm_head.state_dict())
        for param in kg_maskhead_layer.parameters():

            param.requires_grad = False

        self.last=torch.nn.ModuleList()
        for t_indx, _ in self.taskcla:
            if self.qa_task and t_indx in self.qa_task.keys():  # qa task
                self.last.append(torch.nn.Linear(args.roberta_hidden_size, 1))
            else:  # kg task
                self.last.append(kg_maskhead_layer)

        self.args = args
        self.config = config
        self.num_task = len(taskcla)
        self.num_kernel = 3

        print('ROBERTA QA DRAGON ADAPTER CAPSULE MASK')

        return

    def forward(self, t=None, input_ids=None, input_mask=None, s=1):


        output_dict = \
            self.roberta(input_ids=input_ids, attention_mask=input_mask,
                      t=t, s=s)

        sequence_output, pooled_output = output_dict['outputs']

        pooled_output = self.dropout(pooled_output)
        y=[]
        for t_x, _ in self.taskcla:
            if self.qa_task and t_x in self.qa_task.keys() and self.qa_task[t_x]['data_name'] in ['CSQA','OBQA']:
                y.append(self.last[t_x](pooled_output))

            else:
                y.append(self.last[t_x](sequence_output))

        masks = self.mask(t,s)

        output_dict['y'] = y
        output_dict['masks'] = masks

        return output_dict

    def mask(self,t,s):
        masks = {}
        for layer_id in range(self.config.num_hidden_layers):

            if self.args.apply_roberta_output:
                fc1_key = 'roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1' #gfc1
                fc2_key = 'roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2' #gfc2

                masks[fc1_key],masks[fc2_key] = self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.mask(t,s)

                key = 'roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger' #gfc1
                masks[key] = self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.mask(t,s)

            if self.args.apply_roberta_attention_output:
                fc1_key = 'roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1' #gfc1
                fc2_key = 'roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2' #gfc2

                masks[fc1_key],masks[fc2_key] = self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.mask(t,s)

                key = 'roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger' #gfc1
                masks[key] = self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.mask(t,s)

        return masks




    def get_view_for(self,n,p,masks):
        for layer_id in range(self.config.num_hidden_layers):
            if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.weight':
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc1.bias':
                return masks[n.replace('.bias','')].data.view(-1)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.weight':
                post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
                return torch.min(post,pre)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.fc2.bias':
                return masks[n.replace('.bias','')].data.view(-1)

            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.weight':
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc1.bias':
                return masks[n.replace('.bias','')].data.view(-1)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.weight':
                post=masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
                pre=masks[n.replace('.weight','').replace('fc2','fc1')].data.view(1,-1).expand_as(p)
                return torch.min(post,pre)
            elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.fc2.bias':
                return masks[n.replace('.bias','')].data.view(-1)

            if n == \
            'roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.weight': #gfc1
                # print('tsv_capsules not none')
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n == \
            'roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.bias': #gfc1
                return masks[n.replace('.bias','')].data.view(-1)

            if n == \
            'roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.weight': #gfc1
                # print('tsv_capsules not none')
                return masks[n.replace('.weight','')].data.view(-1,1).expand_as(p)
            elif n == \
            'roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.larger.bias': #gfc1
                return masks[n.replace('.bias','')].data.view(-1)

        return None

    def get_view_for_tsv(self,n,t):
        for layer_id in range(self.config.num_hidden_layers):
            if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights':
                # print('not none')
                return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)
            for c_t in range(self.num_task):
                if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias':
                    return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight':
                    # print('not none')
                    return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias':
                    return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.bias':
                    return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                for m_t in range(self.num_kernel):
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.bias':
                        # print('not none')
                        return self.roberta.encoder.layer[layer_id].output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

            if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.route_weights':
                # print('not none')
                return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t].data.view(1,-1,1,1)

            for c_t in range(self.num_task):
                if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc1.'+str(c_t)+'.bias':
                    return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.weight':
                    # print('not none')
                    return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.semantic_capsules.fc2.'+str(c_t)+'.bias':
                    return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

                if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.weight':
                    # print('attention semantic_capsules fc1')
                    return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                elif n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.fc_aspect.'+str(c_t)+'.bias':
                    return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data


                for m_t in range(self.num_kernel):
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs3.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs2.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.weight':
                        return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data
                    if n=='roberta.encoder.layer.'+str(layer_id)+'.attention.output.adapter_capsule_mask.capsule_net.transfer_capsules.convs1.'+str(c_t)+'.'+str(m_t)+'.bias':
                        return self.roberta.encoder.layer[layer_id].attention.output.adapter_capsule_mask.capsule_net.tsv_capsules.tsv[t][c_t].data

        return 1 #if no condition is satified