import sys

from tqdm import tqdm
import numpy as np
import torch

import utils


sys.path.append("./approaches/base/")

from bert_adapter_mask_base import Appr as ApprBase
from my_optimization import BertAdam




class Appr(ApprBase):


    def __init__(self,model,logger,taskcla, qa_task=None, args=None):
        super().__init__(model=model,logger=logger,taskcla=taskcla,args=args)

        self.qa_task = qa_task
        self.qa_ce = torch.nn.CrossEntropyLoss(reduction='mean')
        print('DIL ROBERTA ADAPTER MASK SUP NCL')

        return


    def train(self, t, train, valid, test, num_train_steps, train_data, valid_data):

        if t in self.qa_task.keys():
            if self.qa_task[t]['data_name'] in ['CSQA', 'OBQA']:
                self.train_csqa(t, train, valid, num_train_steps)
        else:
            self.train_mlm(t, train, valid, num_train_steps)

        return


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
            global_step, train_loss = self.train_epoch_mlm(t,train,iter_bar, optimizer,t_total,global_step,e)


            valid_loss = self.eval_mlm(t,valid,trained_task=t)


            # Adapt lr
            if valid_loss<best_loss:
                best_loss=valid_loss
                best_dev_epoch = e
                self.model.to('cpu')
                best_model=utils.get_model(self.model)
                self.model.to(self.device)
                self.logger.info(' *')
                print(' *',end='')

            if e - best_dev_epoch >= self.args.max_epochs_before_stop_graph:
                print('BEST EPOCH:', str(best_dev_epoch), 'Acc: ', str(best_loss))
                break



        # Restore best
        self.model.to('cpu')
        utils.set_model_(self.model,best_model)
        self.model.to(self.device)


        if self.args.multi_gpu and not self.args.distributed:
            mask=self.model.mask(t,s=self.smax)
        else:
            mask=self.model.mask(t,s=self.smax)
        for key,value in mask.items():
            mask[key]=torch.autograd.Variable(value.data.clone(),requires_grad=False)

        if t==0:
            self.mask_pre=mask
        else:
            for key,value in self.mask_pre.items():
                self.mask_pre[key]=torch.max(self.mask_pre[key],mask[key])


        # Weights mask
        self.mask_back={}
        for n,p in self.model.named_parameters():
            if self.args.multi_gpu and not self.args.distributed:
                # vals=self.model.module.get_view_for(n,p,self.mask_pre)
                vals=self.model.get_view_for(n,p,self.mask_pre)
            else:
                vals=self.model.get_view_for(n,p,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals


        return

    def train_epoch_mlm(self,t,data,iter_bar,optimizer,t_total,global_step,e):

        self.model.train()
        total_mlm_loss = 0

        for step, batch in enumerate(iter_bar):
            batch_loss = 0

            batch = [
                bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, input_mask, targets, _= batch
            s=(self.smax-1/self.smax)*step/len(data)+1/self.smax

            bs = input_ids.shape[0]
            for a in range(0, bs, self.args.mini_batch_size_graph):
                b = min(a + self.args.mini_batch_size_graph, bs)
                input_ids_mini_batch = input_ids[a:b]
                input_mask_mini_batch = input_mask[a:b]
                targets_mini_batch = targets[a:b]


                # supervised CE loss ===============
                output_dict = self.model(t=t,input_ids=input_ids_mini_batch, input_mask=input_mask_mini_batch, s=s)
                masks = output_dict['masks']
                pooled_rep = output_dict['normalized_pooled_rep']
                if 'dil' in self.args.scenario:
                    output=output_dict['y']
                elif 'til' in self.args.scenario:
                    outputs = output_dict['y']
                    output = outputs[t]

                loss,_ = self.hat_criterion_adapter(output.view(-1, self.model.tokenizer.vocab_size), targets_mini_batch.view(-1), masks)
                loss = loss * (b - a) / bs
                loss.backward()
                batch_loss += loss.item()

            total_mlm_loss += batch_loss

            # transfer contrastive ===============

            if self.args.amix and t > 0:
                loss += self.amix_loss(output,pooled_rep,input_ids, input_mask,targets, t,s)

            if self.args.augment_distill and t > 0: #separatlu append
                loss += self.augment_distill_loss(output,pooled_rep,input_ids, input_mask,targets, t,s)

            if self.args.sup_loss:
                loss += self.sup_loss(output,pooled_rep,input_ids,  input_mask,targets,t,s)


            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())

            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step

            torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

            # break
        return global_step, (total_mlm_loss/(step+1))





    def train_csqa(self,t, train, valid, num_train_steps):

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

        best_loss=0
        best_dev_epoch = 0
        self.model.to('cpu')
        best_model=utils.get_model(self.model)
        self.model.to(self.device)

        # Loop epochs
        for e in range(int(self.args.num_train_epochs_qa)):
            # Train
            iter_bar = tqdm(train, desc='Train Iter (loss=X.XXX)')
            global_step, train_loss = self.train_epoch_csqa(t,train,iter_bar, optimizer, t_total,global_step)

            valid_loss, valid_acc = self.eval_csqa(t,valid, use_ce_loss=True)



            if valid_acc > best_loss:
                best_loss = valid_acc
                best_dev_epoch = e
                self.model.to('cpu')
                best_model = utils.get_model(self.model)
                self.model.to(self.device)
                print(' *', end='')

            if e - best_dev_epoch >= self.args.max_epochs_before_stop:
                print('BEST EPOCH:',str(best_dev_epoch), 'Acc: ', str(best_loss))
                break





        # Restore best
        self.model.to('cpu')
        utils.set_model_(self.model, best_model)
        self.model.to(self.device)



        if self.args.multi_gpu and not self.args.distributed:
            mask = self.model.mask(t, s=self.smax)
        else:
            mask = self.model.mask(t, s=self.smax)
        for key, value in mask.items():
            mask[key] = torch.autograd.Variable(value.data.clone(), requires_grad=False)


        if t == 0:
            self.mask_pre = mask
        else:
            for key, value in self.mask_pre.items():
                self.mask_pre[key] = torch.max(self.mask_pre[key], mask[key])


        # Weights mask
        self.mask_back = {}
        for n, p in self.model.named_parameters():
            if self.args.multi_gpu and not self.args.distributed:
                vals = self.model.get_view_for(n, p, self.mask_pre)
            else:
                vals = self.model.get_view_for(n, p, self.mask_pre)
            if vals is not None:
                self.mask_back[n] = 1 - vals

        return

    def train_epoch_csqa(self, t, data, iter_bar, optimizer, t_total, global_step):
        self.model.train()
        total_loss_during_taining = 0
        for step, batch in enumerate(iter_bar):
            batch_loss = 0
            batch = [bat.to(self.device) if bat is not None else None for bat in batch]
            input_ids, input_mask, targets, _ = batch
            s = (self.smax - 1 / self.smax) * step / len(data) + 1 / self.smax

            bs = input_ids.shape[0]
            for a in range(0, bs, self.args.mini_batch_size):
                b = min(a + self.args.mini_batch_size, bs)
                input_ids_mini_batch = input_ids[a:b]
                input_mask_mini_batch = input_mask[a:b]
                targets_mini_batch = targets[a:b]

                input_ids_reshaped = input_ids_mini_batch.view(input_ids_mini_batch.shape[0] * input_ids_mini_batch.shape[1], -1)
                input_mask_reshaped = input_mask_mini_batch.view(input_mask_mini_batch.shape[0] * input_mask_mini_batch.shape[1], -1)

                output_dict = self.model.forward(t=t, input_ids=input_ids_reshaped, input_mask=input_mask_reshaped,s=s)
                outputs = output_dict['y']
                masks = output_dict['masks']
                pooled_rep = output_dict['normalized_pooled_rep']
                # Forward
                output = outputs[t]
                reshaped_output = output.view(int((output.shape[0] / self.qa_task[t]['n_classes'])),
                                            self.qa_task[t]['n_classes'])

                loss = self.qa_ce(reshaped_output, targets_mini_batch)
                loss = loss * (b - a) / bs
                loss.backward()
                batch_loss += loss.item()


            total_loss_during_taining += batch_loss


            # Restrict layer gradients in backprop
            if t>0:
                for n,p in self.model.named_parameters():
                    if n in self.mask_back:
                        p.grad.data*=self.mask_back[n]

            # Compensate embedding gradients
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                    den=torch.cosh(p.data)+1
                    p.grad.data*=self.smax/s*num/den

            lr_this_step = self.args.learning_rate * \
                           self.warmup_linear(global_step/t_total, self.args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step



            torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clipgrad)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1


            iter_bar.set_description('Train Iter (loss=%5.3f)' % loss.item())


            # Constrain embeddings
            for n,p in self.model.named_parameters():
                if 'adapter_mask.e' in n or n.startswith('e'):
                    p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)


        return global_step, (total_loss_during_taining/(step+1))




    def eval_mlm(self,t,data,test=None,trained_task=None, filter_eval=False):
        total_loss=0
        total_num=0

        # for acc evaluation
        total_acc = 0
        target_list = []
        pred_list = []

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


                output_dict=self.model.forward(t=t, input_ids=input_ids, input_mask=input_mask,s=self.smax)
                outputs = output_dict['y']
                masks = output_dict['masks']
                output = outputs[t]

                # Forward
                loss, _ = self.hat_criterion_adapter(output.view(-1, self.model.tokenizer.vocab_size), targets.view(-1),
                                                     masks)


                pred = output.argmax(2)[(targets != -100).nonzero(as_tuple=True)]
                targets_for_acc = targets[(targets != -100).nonzero(as_tuple=True)]
                hits = (pred == targets_for_acc).float()
                total_acc += hits.sum().data.cpu().numpy().item()


                total_loss+=loss.data.cpu().numpy().item()*real_b
                total_num+=real_b


        print('MLM Accuracy: ', total_acc/total_num)
        return total_loss/total_num


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
                output_dict = self.model.forward(t=t, input_ids=input_ids_reshaped, input_mask=input_mask_reshaped,s=self.smax)
                outputs = output_dict['y']
                masks = output_dict['masks']
                output = outputs[t]
                reshaped_output = output.view(int((output.shape[0] / self.qa_task[t]['n_classes'])),
                                              self.qa_task[t]['n_classes'])
                total_num += real_b

                if use_ce_loss:  # validation returns crossentropy
                    # loss = self.ce(reshaped_output, targets)  # crossentropy
                    loss, _ = self.hat_criterion_adapter(reshaped_output,
                                                         targets,
                                                         masks)
                    total_loss += loss.data.cpu().numpy().item() * real_b
                    # for hits
                    predicted_answer = reshaped_output.argmax(dim=1)
                    hits = (predicted_answer == targets).sum().item()
                    total_acc += hits

                else:  # test saves predictions to json file
                    predicted_answer = reshaped_output.argmax(dim=1)
                    hits = (predicted_answer == targets).sum().item()
                    total_acc += hits



            if use_ce_loss:
                return total_loss / total_num, total_acc / total_num
            else:
                print('total_acc/total_num: ', total_acc / total_num)
                return total_acc / total_num


    def eval(self, t, data, use_ce_loss=False, data_path=None, filter_eval=False):
        if t in self.qa_task.keys():
            if self.qa_task[t]['data_name'] in ['CSQA', 'OBQA']:
                loss_value = self.eval_csqa(t, data, use_ce_loss, data_path)
        else:
            use_ce_loss = True
            loss_value = self.eval_mlm(t, data, use_ce_loss, data_path, filter_eval=filter_eval)

        return loss_value
