#coding: utf-8
import sys
import torch
from transformers import RobertaModel, RobertaConfig
import utils
from torch import nn
import torch.nn.functional as F
sys.path.append("./networks/base/")
from my_roberta_transformers import MyRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaForMaskedLM
from transformers import RobertaTokenizer


class Net(torch.nn.Module):
# sinngle dapater for all task fir iterative training
    def __init__(self, taskcla, qa_task, args):

        super(Net,self).__init__()
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        args.build_adapter = True
        self.roberta = MyRobertaModel.from_pretrained(args.bert_model,config=config,args=args)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)
        self.qa_task = qa_task
        #BERT fixed all ===========
        for param in self.roberta.parameters():
            param.requires_grad = False



        #Only adapters are trainable

        if args.apply_roberta_output and args.apply_roberta_attention_output:
            adaters = \
                [self.roberta.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif args.apply_roberta_output:
            adaters = \
                [self.roberta.encoder.layer[layer_id].output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in range(config.num_hidden_layers)]

        elif self.apply_roberta_attention_output:
            adaters = \
                [self.roberta.encoder.layer[layer_id].attention.output.adapter for layer_id in range(config.num_hidden_layers)] + \
                [self.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in range(config.num_hidden_layers)]


        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = True
                # param.requires_grad = False

        self.taskcla=taskcla
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        # init MLMHead, HF Work around
        original_model_for_mask_pred = RobertaForMaskedLM.from_pretrained(args.bert_model)
        kg_maskhead_layer = RobertaLMHead(config)
        kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.lm_head.state_dict())
        for param in kg_maskhead_layer.parameters():
            # param.requires_grad = True
            param.requires_grad = False
        self.args = args
        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.roberta_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            self.last=torch.nn.ModuleList()
            for t_indx, _ in self.taskcla:
                if self.qa_task and t_indx in self.qa_task.keys(): # qa task
                    self.last.append(torch.nn.Linear(args.roberta_hidden_size, 1))
                else: # knowledge task, reuse mlm task
                    self.last.append(kg_maskhead_layer)

        print('ROBERTA SINGLE ADAPTER FOR ALL TASKS')

        return

    def forward(self, input_ids, input_mask):
        output_dict = {}


        sequence_output, pooled_output = self.roberta(input_ids=input_ids, attention_mask=input_mask)
        pooled_output = self.dropout(pooled_output)
        #shared head

        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            for t_num, _ in self.taskcla:
                if self.qa_task and t_num in self.qa_task.keys() and self.qa_task[t_num]['data_name'] in ['CSQA', 'OBQA']:
                    y.append(self.last[t_num](pooled_output))
                else:
                    y.append(self.last[t_num](sequence_output))

        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)

        return output_dict