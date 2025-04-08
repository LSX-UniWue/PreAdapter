#coding: utf-8
import sys
import torch
from transformers import RobertaModel, RobertaConfig
import utils
from torch import nn
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaForMaskedLM
from transformers import RobertaTokenizer

class Net(torch.nn.Module):

    def __init__(self,taskcla, qa_task, args):

        super(Net,self).__init__()
        config = RobertaConfig.from_pretrained(args.bert_model)
        config.return_dict=False
        self.args = args
        self.qa_task = qa_task
        self.roberta = RobertaModel.from_pretrained(args.bert_model,config=config)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)


        self.taskcla=taskcla
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


        if 'dil' in args.scenario:
            self.last=torch.nn.Linear(args.roberta_hidden_size,args.nclasses)
        elif 'til' in args.scenario:
            # init MLMHead, HF Work around
            original_model_for_mask_pred = RobertaForMaskedLM.from_pretrained(args.bert_model)
            # freeze and reuse classification layer
            kg_maskhead_layer = RobertaLMHead(config)
            kg_maskhead_layer.load_state_dict(original_model_for_mask_pred.lm_head.state_dict())
            # freeze and reuse mask layer
            for param in kg_maskhead_layer.parameters():
                # param.requires_grad = True
                param.requires_grad = False
            self.last=torch.nn.ModuleList()
            for t_indx, _ in self.taskcla:
                if self.qa_task and t_indx in self.qa_task.keys(): # qa task
                    self.last.append(torch.nn.Linear(args.roberta_hidden_size, 1))
                else: # knowledge task, reuse mlm task
                    self.last.append(kg_maskhead_layer)
        print('ROBERTA')

        return

    def forward(self, t=None, input_ids=None, input_mask=None):
        output_dict = {}

        sequence_output, pooled_output = \
            self.roberta(input_ids=input_ids, attention_mask=input_mask)

        pooled_output = self.dropout(pooled_output)

        if 'dil' in self.args.scenario:
            y = self.last(pooled_output)
        elif 'til' in self.args.scenario:
            y=[]
            if self.qa_task and t in self.qa_task.keys() and self.qa_task[t]['data_name'] in ['CSQA', 'OBQA']:
                y.append(self.last[t](pooled_output))

            else:
                y.append(self.last[t](sequence_output)) # for mlm task, knowledge integration


        output_dict['y'] = y
        output_dict['normalized_pooled_rep'] = F.normalize(pooled_output, dim=1)

        return output_dict