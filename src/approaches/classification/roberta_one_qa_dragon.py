import sys,time
import numpy as np
import torch
import os
import logging
import glob
import math
import json
import argparse
import random
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.data import TensorDataset, random_split
import utils
# from apex import amp
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam

from copy import deepcopy
sys.path.append("./approaches/base/")
from optimization_utils import OPTIMIZER_CLASSES
from transformers.optimization import get_constant_schedule # WarmupLinearSchedule, WarmupConstantSchedule,
from bert_base import Appr as ApprBase
from datetime import datetime
class Appr(ApprBase):



    def __init__(self,model,logger, taskcla, qa_task=None, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        self.qa_task = qa_task
        self.qa_ce = torch.nn.CrossEntropyLoss(reduction='mean')
        print('ROBERTA ONE')


        return


    def train(self, t, train, valid, test, num_train_steps, train_data, valid_data):

        if self.qa_task and t in self.qa_task.keys():
            if self.qa_task[t]['data_name'] in ['CSQA', 'OBQA']:
                self.train_csqa(t, train, valid, num_train_steps)
        else:
            self.train_mlm(t, train, valid, num_train_steps)

        return

    def train_csqa(self,t, train, valid, num_train_steps):

        global_step = 0
        self.model.to(self.device)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': self.args.encoder_lr, 'weight_decay': 0.01},#self.args.weight_decay
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': self.args.encoder_lr, 'weight_decay': 0.0}
        ]

        optimizer = OPTIMIZER_CLASSES['radam'](optimizer_grouped_parameters)
        scheduler = get_constant_schedule(optimizer)

        t_total = num_train_steps

        best_loss=0
        best_dev_epoch = 0
        self.model.to('cpu')
        best_model=utils.get_model(self.model)
        self.model.to(self.device)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs_qa)):
            # Train

            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step, train_loss =self.train_epoch_csqa(t,train,iter_bar, optimizer, scheduler, t_total,global_step)


            valid_loss, valid_loss_acc = self.eval_csqa(t,valid, use_ce_loss=True)




            if valid_loss_acc > best_loss:
                best_loss = valid_loss_acc
                best_dev_epoch = e
                self.model.to('cpu')
                best_model = utils.get_model(self.model)
                self.model.to(self.device)
                print(' *', end='')

            if e - best_dev_epoch >= self.args.max_epochs_before_stop:
                print('BEST EPOCH:',str(e), 'Acc: ', str(best_loss))
                break

        # Restore best
        self.model.to('cpu')
        utils.set_model_(self.model, best_model)
        self.model.to(self.device)

        return

    def train_epoch_csqa(self, t, data, iter_bar, optimizer, scheduler, t_total, global_step):
        self.model.train()
        total_loss_during_taining = 0
        for step, batch in enumerate(iter_bar):
            batch_loss = 0
            batch = [bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, input_mask, targets, _ = batch
            bs = input_ids.shape[0]
            for a in range(0, bs, self.args.mini_batch_size):
                b = min(a + self.args.mini_batch_size, bs)
                input_ids_mini_batch = input_ids[a:b]
                input_mask_mini_batch = input_mask[a:b]
                targets_mini_batch = targets[a:b]
                input_ids_reshaped = input_ids_mini_batch.view(input_ids_mini_batch.shape[0] * input_ids_mini_batch.shape[1], -1)
                input_mask_reshaped = input_mask_mini_batch.view(input_mask_mini_batch.shape[0] * input_mask_mini_batch.shape[1], -1)

                output_dict = self.model.forward(t=t, input_ids=input_ids_reshaped, input_mask=input_mask_reshaped)
                outputs = output_dict['y']
                pooled_rep = output_dict['normalized_pooled_rep']
                # Forward
                output = outputs[0]
                reshaped_output = output.view(int((output.shape[0] / self.qa_task[t]['n_classes'])),
                                          self.qa_task[t]['n_classes'])
                loss = self.qa_ce(reshaped_output, targets_mini_batch)
                loss = loss * (b - a) / bs
                loss.backward()
                batch_loss += loss.item()

            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            iter_bar.desc = "loss: {:.2e}  lr: {:.2e}".format(batch_loss, scheduler.get_lr()[0])

            total_loss_during_taining += batch_loss

            optimizer.zero_grad()
            global_step += 1

        return global_step, (total_loss_during_taining/(step+1))

    def eval_csqa(self, t, data, test=None, trained_task=None, filter_eval=False, use_ce_loss=False):
        total_loss = 0
        total_num = 0
        total_acc = 0
        self.model.eval()

        iter_bar = tqdm(data, desc='Eval Iter (loss=X.XXX)')
        with torch.no_grad():

            for step, batch in enumerate(iter_bar):
                batch = [bat.to(self.device) if bat is not None else None for bat in batch]
                input_ids, input_mask, targets, _ = batch

                input_ids_reshaped = input_ids.view(input_ids.shape[0] * input_ids.shape[1], -1)
                input_mask_reshaped = input_mask.view(input_mask.shape[0] * input_mask.shape[1], -1)


                real_b = input_ids.size()[0]

                # Forward
                output_dict = self.model.forward(t=t, input_ids=input_ids_reshaped, input_mask=input_mask_reshaped)
                outputs = output_dict['y']
                output = outputs[0]
                reshaped_output = output.view(int((output.shape[0] / self.qa_task[t]['n_classes'])),
                                              self.qa_task[t]['n_classes'])
                total_num += real_b

                if use_ce_loss:  # validation returns crossentropy
                    loss = self.qa_ce(reshaped_output, targets)  # crossentropy
                    total_loss += loss.data.cpu().numpy().item() * real_b
                    predicted_answer = reshaped_output.argmax(dim=1)
                    hits = (predicted_answer == targets).sum().item()
                    total_acc += hits


                else:
                    predicted_answer = reshaped_output.argmax(dim=1)
                    hits = (predicted_answer == targets).sum().item()
                    total_acc += hits

            if use_ce_loss:
                return total_loss / total_num, total_acc / total_num
            else:
                print('total_acc/total_num: ', total_acc / total_num)
                return total_acc / total_num

    def train_mlm(self, t, train, valid, num_train_steps):

        global_step = 0
        self.model.to(self.device)
        param_optimizer = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad==True]
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        t_total = num_train_steps
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=self.args.learning_rate,
                             warmup=self.args.warmup_proportion,
                             t_total=t_total)


        best_loss=np.inf
        best_dev_epoch = 0
        self.model.to('cpu')
        best_model=utils.get_model(self.model)
        self.model.to(self.device)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs)):
            # Train

            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step, train_loss =self.train_epoch_mlm(t,train,iter_bar, optimizer,t_total,global_step)


            valid_loss = self.eval_mlm(t,valid)
            print(' Valid: loss={:.3f}|'.format(valid_loss),end='')



            if valid_loss<best_loss:
                best_loss=valid_loss
                best_dev_epoch = e
                best_model_path = self.args.model_save_path + 'best_model_got_after_epoch_' + str(e) + '_for_task_'+str(t)+'_' + datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S")
                torch.save(self.model, best_model_path)
                torch.save(self.model.state_dict(), best_model_path+'_state_dict')
                self.model.to('cpu')
                best_model=utils.get_model(self.model)
                self.model.to(self.device)
                print(' *',end='')

            if e - best_dev_epoch >= self.args.max_epochs_before_stop_graph:
                print('BEST EPOCH:', str(e), 'Acc: ', str(best_loss))
                break


        # Restore best
        self.model.to('cpu')
        utils.set_model_(self.model,best_model)
        self.model.to(self.device)

        return


    def train_epoch_mlm(self,t,data,iter_bar,optimizer,t_total,global_step):
        self.model.train()
        total_mlm_loss = 0
        for step, batch in enumerate(iter_bar):
            batch_loss = 0
            # print('step: ',step)
            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, input_mask, targets, _ = batch
            bs = input_ids.shape[0]
            for a in range(0, bs, self.args.mini_batch_size_graph):
                b = min(a + self.args.mini_batch_size_graph, bs)
                input_ids_mini_batch = input_ids[a:b]
                input_mask_mini_batch = input_mask[a:b]
                targets_mini_batch = targets[a:b]

                output_dict = self.model.forward(t=t, input_ids=input_ids_mini_batch, input_mask=input_mask_mini_batch)
                outputs = output_dict['y']
                pooled_rep = output_dict['normalized_pooled_rep']
                # Forward
                output = outputs[0]
                # loss=self.ce(output,targets)
                loss= self.ce(output.view(-1, self.model.tokenizer.vocab_size), targets_mini_batch.view(-1))
                loss = loss * (b - a) / bs
                loss.backward()
                batch_loss += loss.item()

            total_mlm_loss += batch_loss

            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())


            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        return global_step, (total_mlm_loss/(step+1))

    def eval_mlm(self,t,data,test=None,trained_task=None, filter_eval=False):
        total_loss=0
        total_num=0

        self.model.eval()

        with torch.no_grad():
            iter_bar = tqdm(data, desc='Eval Iter (loss=X.XXX)')
            for step, batch in enumerate(iter_bar):
                batch = [
                    bat.to(self.device) if bat is not None else None for bat in batch]
                if filter_eval:
                    input_ids, input_mask, targets, _, vocab_vector = batch
                else:
                    input_ids, input_mask, targets, _= batch
                real_b=input_ids.size(0)

                output_dict = self.model.forward(t=t, input_ids=input_ids, input_mask=input_mask)
                outputs = output_dict['y']

                # Forward
                output = outputs[0]
                loss=self.ce(output.view(-1, self.model.tokenizer.vocab_size),targets.view(-1))

                # Log
                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b

        return total_loss / total_num

    def eval(self, t, data, use_ce_loss=False, data_path=None, filter_eval=False):
        if self.qa_task and t in self.qa_task.keys():
            if self.qa_task[t]['data_name'] in ['CSQA', 'OBQA']:
                loss_value = self.eval_csqa(t, data, use_ce_loss, data_path)
        else:
            use_ce_loss = True
            loss_value = self.eval_mlm(t, data, use_ce_loss, data_path)

        return loss_value