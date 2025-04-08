from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from transformers import RobertaTokenizer as RobertaTokenizer
import os
import torch
import numpy as np
import random
import nlp_data_utils as data_utils
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import math
from datasets import load_dataset



domains = ['CN_OBQA', 'CN_CSQA', 'CSQA', 'OBQA',]
knowledge_domains = ['CN_OBQA', 'CN_CSQA']
qa_domains = ['CSQA', 'OBQA']

datasets = ['./dat/qa_conceptNet/' + domain for domain in domains]


def get(logger=None, args=None):

    data = {}
    taskcla = []
    qa_task = {}

    f_name = './data_prep/qa_cn_' + str(args.ntasks)
    if not os.path.isfile(f_name): f_name = './data_prep/qa_cn'
    print('### LOAD Data From:', f_name)
    print(f_name)
    with open(f_name, 'r') as f_random_seq:
        random_sep = f_random_seq.readlines()[args.idrandom].split()
        # random.shuffle(random_sep)
    print('random_sep: ', random_sep)
    print('domains: ', domains)
    seq_inf = {}
    seq_inf['seq_file_name'] = f_name
    for t in range(args.ntasks):
        dataset = datasets[domains.index(random_sep[t])]
        rel_type = random_sep[t]
        seq_inf[rel_type] = t
        data[t] = {}
        data[t]['name'] = dataset
        data[t]['ncla'] = 2

        print('dataset: ', dataset)
        logger.info(dataset)

        if random_sep[t] in ['CSQA']: # read CSQA question, answers for Multiple Choice Question Answering
            processor = data_utils.CSQA_BERT_Processor()
        elif random_sep[t] == 'OBQA': # read OBQA question, answers for Multiple Choice Question Answering
            processor = data_utils.OBQA_BERT_Processor()
        elif random_sep[t] in ['CN_CSQA', 'CN_OBQA']:
            processor = data_utils.CN_RoBERTa_Processor()
        else:
            processor = None
        tokenizer = RobertaTokenizer.from_pretrained(args.bert_model)  # tokenizer
        label_list = processor.get_labels(model_tokenizer=tokenizer)
        data[t]['ncla'] = label_list

        print("### GET TRAIN EXAMPLES ###")
        if random_sep[t] in ['CN_CSQA', 'CN_OBQA']:
            train_examples = processor.get_train_examples(dataset, debug=args.debug,
                                                          model_tokenizer=tokenizer,
                                                          triple_representation=args.triple_representation,
                                                          max_len=args.max_seq_length,
                                                          args=args)
        else:
            train_examples = processor.get_train_examples(dataset, debug=args.debug,
                                                          model_tokenizer=tokenizer,
                                                          triple_representation=args.triple_representation,
                                                          max_len=args.max_seq_length)


        if args.train_data_size > 0:
            random.Random(args.data_seed).shuffle(train_examples)  # more robust
            border = min(len(train_examples),args.train_data_size)
            train_examples = train_examples[:border]



        if random_sep[t] in ['CSQA', 'OBQA']:
            adjusted_batch_size = math.floor(args.train_batch_size / len(train_examples[0].choices)) # n answers per 1 question
            num_train_steps = int(math.ceil(len(train_examples) / adjusted_batch_size)) * args.num_train_epochs_qa
        else:
            num_train_steps = int(math.ceil(len(train_examples) / args.train_batch_size)) * args.num_train_epochs


        if random_sep[t] in ['CSQA', 'OBQA']:
            train_features = data_utils.convert_examples_to_features_csqa_bert(
                train_examples, args.max_seq_length, tokenizer)
        elif random_sep[t] in ['CN_OBQA', 'CN_CSQA']:
            train_features = data_utils.convert_examples_to_features_cn_bert(
                train_examples, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_tasks)

        data[t]['train'] = train_data
        data[t]['num_train_steps'] = num_train_steps


        if random_sep[t] in qa_domains:
            data[t]['ncla'] = len(train_features[0].input_ids)

        print('### GET VALID EXAMPLES ###')
        if random_sep[t] in ['CN_CSQA','CN_OBQA']:
            valid_examples = processor.get_dev_examples(dataset, debug=args.debug,
                                                        model_tokenizer=tokenizer,
                                                        max_len=args.max_seq_length,
                                                        args=args)
        else:
            valid_examples = processor.get_dev_examples(dataset, debug=args.debug,
                                                        model_tokenizer=tokenizer,
                                                        max_len=args.max_seq_length)

        if args.dev_data_size > 0:
            random.Random(args.data_seed).shuffle(valid_examples)  # more robust
            border = min(len(valid_examples), args.dev_data_size)
            valid_examples = valid_examples[:border]



        if random_sep[t] in ['CSQA','OBQA']:
            valid_features = data_utils.convert_examples_to_features_csqa_bert(
                valid_examples, args.max_seq_length, tokenizer)
        elif random_sep[t] in ['CN_OBQA', 'CN_CSQA']:
            valid_features = data_utils.convert_examples_to_features_cn_bert(
                valid_examples, args.max_seq_length, tokenizer)
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_label_ids = torch.tensor([f.label_id for f in valid_features], dtype=torch.long)
        valid_all_tasks = torch.tensor([t for f in valid_features], dtype=torch.long)

        valid_data = TensorDataset(valid_all_input_ids, valid_all_input_mask,
                                   valid_all_label_ids, valid_all_tasks)

        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)

        data[t]['valid'] = valid_data

        if random_sep[t] in ['CSQA']:  # read CSQA question, answers for Multiple Choice Question Answering
            processor = data_utils.CSQA_BERT_Processor()
        elif random_sep[t] == 'OBQA':
            processor = data_utils.OBQA_BERT_Processor()


        print('### GET TEST EXAMPLES ###')
        if random_sep[t] in ['CN_CSQA', 'CN_OBQA']:
            eval_examples = processor.get_test_examples(dataset, debug=args.debug, model_tokenizer=tokenizer,
                                                        max_len=args.max_seq_length, args=args, isTest=True)
        else:
            eval_examples = processor.get_test_examples(dataset, debug=args.debug, model_tokenizer=tokenizer,
                                                    max_len=args.max_seq_length, isTest=True)

        if args.test_data_size > 0:  # TODO: COMMENT OUT
            random.Random(args.data_seed).shuffle(eval_examples)  # more robust
            border = min(len(eval_examples), args.test_data_size)
            eval_examples = eval_examples[:border]


        if random_sep[t] in ['CSQA','OBQA']:
            eval_features = data_utils.convert_examples_to_features_csqa_bert(eval_examples,
                                                                               args.max_seq_length, tokenizer)
        elif random_sep[t] in ['CN_OBQA', 'CN_CSQA']:
            eval_features = data_utils.convert_examples_to_features_cn_bert(eval_examples,
                                                                                args.max_seq_length, tokenizer)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_tasks = torch.tensor([t for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids, all_tasks)  # all_possible_labels)
        # Run prediction for full data

        data[t]['test'] = eval_data


        taskcla.append((t, int(data[t]['ncla'])))
        if random_sep[t] in qa_domains:
            qa_task[t] = {'data_name': random_sep[t], 'n_classes': len(train_features[0].input_ids)}

    # Others
    n = 0
    for t in data.keys():
        n += data[t]['ncla']
    data['ncla'] = n
    data['task_seq_inf'] = seq_inf

    return data, taskcla, qa_task
