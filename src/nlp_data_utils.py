# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json, ijson
import os
import numpy as np
from transformers import BertTokenizer, BartTokenizer, RobertaTokenizer
import string
import torch
from random import randint, shuffle, choice
from random import random as rand
import random as rnd
from config import set_args
import transformers
from datasets import load_dataset
import random
import csv
from typing import List
import re

transformer_args = set_args()


class ABSATokenizer(BertTokenizer):
    def subword_tokenize(self, tokens, labels): # for AE
        split_tokens, split_labels= [], []
        idx_map=[]
        for ix, token in enumerate(tokens):
            sub_tokens=self.wordpiece_tokenizer.tokenize(token)
            for jx, sub_token in enumerate(sub_tokens):
                split_tokens.append(sub_token)
                if labels[ix]=="B" and jx>0:
                    split_labels.append("I")
                else:
                    split_labels.append(labels[ix])
                idx_map.append(ix)
        return split_tokens, split_labels, idx_map





class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text_a=None, text_b=None, label=None, relation_type=None, e1=None, e2=None, all_possible_labels=None, weight=None, choices=None, max_pair_len=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.relation_type = relation_type
        self.e1 = e1
        self.e2 = e2
        self.all_possible_labels = all_possible_labels
        self.weight = weight
        self.choices = choices
        self.max_pair_len = max_pair_len


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                    input_ids=None,
                    input_mask=None,
                    segment_ids=None,

                    tokens_term_ids=None,
                    tokens_sentence_ids=None,

                    term_input_ids=None,
                    term_input_mask=None,
                    term_segment_ids=None,

                    sentence_input_ids=None,
                    sentence_input_mask=None,
                    sentence_segment_ids=None,

                    tokens_term_sentence_ids=None,
                    label_id=None,

                    masked_lm_labels = None,
                    masked_pos = None,
                    masked_weights = None,

                    position_ids=None,

                    valid_ids=None,
                    label_mask=None,

                    all_possible_labels = None,
                    masked_input_ids=None,
                    input_masks_mlm=None,
                    segment_ids_mlm=None,
                    labels_mlm=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.label_id = label_id

        self.masked_lm_labels = masked_lm_labels,
        self.masked_pos = masked_pos,
        self.masked_weights = masked_weights

        self.tokens_term_ids = tokens_term_ids
        self.tokens_sentence_ids = tokens_sentence_ids

        self.term_input_ids = term_input_ids
        self.term_input_mask = term_input_mask
        self.term_segment_ids = term_segment_ids

        self.sentence_input_ids = sentence_input_ids
        self.sentence_input_mask = sentence_input_mask
        self.sentence_segment_ids = sentence_segment_ids

        self.tokens_term_sentence_ids= tokens_term_sentence_ids

        self.position_ids = position_ids

        self.valid_ids = valid_ids
        self.label_mask = label_mask

        self.all_possible_labels = all_possible_labels

        self.masked_input_ids = masked_input_ids
        self.input_masks_mlm = input_masks_mlm
        self.segment_ids_mlm = segment_ids_mlm
        self.labels_mlm = labels_mlm



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        with open(input_file) as f:
            return json.load(f)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)



class DtcProcessor(DataProcessor):
    """Processor for document text classification."""

    def get_labels(self,ntasks):
        return [t for t in range(ntasks)]

    def _create_examples(self,lines,set_type):
        examples = []
        for i,(sentence,label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = sentence #no need to split for us
            text_b = None
            label = label
            examples.append(InputExample(guid=guid,text_a=text_a,text_b=text_b,label=label))
        return examples

class DscProcessor(DataProcessor):
    """Processor for document text classification."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        return ['-1','1']


    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['sentence']
            text_b = None
            label = lines[ids]['polarity']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



class CSQA_BERT_Processor(DataProcessor):
    """Processor for document text classification."""

    def get_train_examples(self, data_dir, fn="train.jsonl", debug=False, model_tokenizer=None, triple_representation='triple', max_len = 128):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "train", debug=debug, model_tokenizer=model_tokenizer, max_len = max_len)

    def get_dev_examples(self, data_dir, fn="dev.jsonl", debug=False, model_tokenizer=None, labels_dict=None, max_len=128):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "dev", debug=debug, model_tokenizer=model_tokenizer, max_len=max_len)

    def get_test_examples(self, data_dir, fn="test.jsonl", debug=False, model_tokenizer=None, max_len=128, isTest=True):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "test", debug=debug, model_tokenizer=model_tokenizer, max_len=max_len)

    def get_labels(self, model_tokenizer=None):
        # check tokenizer type
        if type(model_tokenizer) == BartTokenizer:
            fb_labels = model_tokenizer.get_vocab().values()
        elif type(model_tokenizer) == RobertaTokenizer:
            fb_labels = model_tokenizer.vocab_size
        else: # BERT
            fb_labels = model_tokenizer.vocab.values()
        return fb_labels

    def _read_large_json(self, input_file):
        with open(input_file, 'rb') as f:
            return f

    # Read prompt templates from a CSV file
    def read_prompt_templates_from_csv(self, file_path: str):
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
        return prompt_templates

    # Create a prompt using the provided data
    def create_question_answer_pairs(self, qa_sample, tokenizer) -> str:
        question = qa_sample['question']['stem']
        # question = 'Q: ' + question + ' A: '
        # print(qa_sample)
        label = qa_sample['answerKey']
        # answer_list = [a['text']+' .' for a in qa_sample['question']['choices']]
        answer_list = [a['text'] for a in qa_sample['question']['choices']]
        qa_pairs = [question + ' ' + a for a in answer_list]
        max_qa_len = max([len(tokenizer.tokenize(qa)) for qa in qa_pairs])
        return question, label, answer_list, max_qa_len


    def _create_examples(self, input_file, set_type, debug=False, model_tokenizer=None, max_len=128):
        """Creates examples for the training and dev sets."""
        # help files: relation templates and relation id -> relation words mapping
        # with open('./dat/iswc_2023_challenge/ISWC/prompts.csv',
        #           'r') as templates:
        examples = []

        with open(input_file, 'r') as f:
            id_c = 0
            # data = json.load(f)
            for line in f:
                line = json.loads(line)
                question, label, choices, max_pair_len = self.create_question_answer_pairs(qa_sample=line, tokenizer=model_tokenizer)
                if debug and id_c > 50:
                    break

                if max_pair_len > (max_len-5):
                    print('too long question: ', question)
                    continue
                guid = "%s-%s" % (set_type, id_c)
                id_c += 1

                examples.append(
                    InputExample(guid=guid, text_a=question, label=label, choices=choices, max_pair_len=max_pair_len))
        print(len(examples))

        return examples



class OBQA_BERT_Processor(DataProcessor):  # MyCode
    """Processor for document text classification."""

    def get_train_examples(self, data_dir, fn="train.jsonl", debug=False, model_tokenizer=None, triple_representation='triple', max_len = 128):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "train", debug=debug, model_tokenizer=model_tokenizer, max_len = max_len)

    def get_dev_examples(self, data_dir, fn="dev.jsonl", debug=False, model_tokenizer=None, labels_dict=None, max_len=128):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "dev", debug=debug, model_tokenizer=model_tokenizer, max_len=max_len)

    def get_test_examples(self, data_dir, fn="test.jsonl", debug=False, model_tokenizer=None, max_len=128, isTest=True):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "test", debug=debug, model_tokenizer=model_tokenizer, max_len=max_len)

    def get_labels(self, model_tokenizer=None):
        # check tokenizer type
        if type(model_tokenizer) == BartTokenizer:
            fb_labels = model_tokenizer.get_vocab().values()
        elif type(model_tokenizer) == RobertaTokenizer:
            fb_labels = model_tokenizer.vocab_size
        else: # BERT
            fb_labels = model_tokenizer.vocab.values()
        return fb_labels

    def _read_large_json(self, input_file):
        with open(input_file, 'rb') as f:
            return f

    # Read prompt templates from a CSV file
    def read_prompt_templates_from_csv(self, file_path: str):
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
        return prompt_templates

    # Create a prompt using the provided data
    def create_question_answer_pairs(self, qa_sample, tokenizer) -> str:
        question = qa_sample['question']['stem']
        # print(qa_sample)
        label = qa_sample['answerKey']
        # answer_list = [a['text']+' .' for a in qa_sample['question']['choices']]
        answer_list = [a['text'] for a in qa_sample['question']['choices']]
        qa_pairs = [question + ' ' + a for a in answer_list]
        max_qa_len = max([len(tokenizer.tokenize(qa)) for qa in qa_pairs])
        return question, label, answer_list, max_qa_len


    def _create_examples(self, input_file, set_type, debug=False, model_tokenizer=None, max_len=128):
        """Creates examples for the training and dev sets."""
        # help files: relation templates and relation id -> relation words mapping
        # with open('./dat/iswc_2023_challenge/ISWC/prompts.csv',
        #           'r') as templates:
        examples = []

        with open(input_file, 'r') as f:
            id_c = 0
            # data = json.load(f)
            for line in f:
                line = json.loads(line)
                question, label, choices, max_pair_len = self.create_question_answer_pairs(qa_sample=line, tokenizer=model_tokenizer)
                if debug and id_c > 50:
                    break

                if max_pair_len > (max_len-4):
                    print('too long question: ', question)
                    continue
                guid = "%s-%s" % (set_type, id_c)
                id_c += 1

                examples.append(
                    InputExample(guid=guid, text_a=question, label=label, choices=choices, max_pair_len=max_pair_len))
        print(len(examples))

        return examples








class CN_RoBERTa_Processor(DataProcessor):
    """Processor for document text classification."""

    def get_train_examples(self, data_dir, fn="train.csv", debug=False, model_tokenizer=None, triple_representation='triple', max_len = 128, args=None):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "train", debug=debug, model_tokenizer=model_tokenizer, max_len = max_len, args=args)

    def get_dev_examples(self, data_dir, fn="dev.csv", debug=False, model_tokenizer=None, labels_dict=None, max_len=128, args=None):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "dev", debug=debug, model_tokenizer=model_tokenizer, max_len=max_len,  args=args)

    def get_test_examples(self, data_dir, fn="test.csv", debug=False, model_tokenizer=None, max_len=128, isTest=True, args=None):
        """See base class."""
        return self._create_examples(os.path.join(data_dir, fn), "test", debug=debug, model_tokenizer=model_tokenizer, max_len=max_len, isTest=True, args=args)

    def get_labels(self, model_tokenizer=None):
        # check tokenizer type
        if type(model_tokenizer) == BartTokenizer:
            labels = model_tokenizer.get_vocab().values()
        elif type(model_tokenizer) == RobertaTokenizer:
            labels = model_tokenizer.vocab_size
        else: # BERT
            labels = model_tokenizer.vocab.values()
        return labels

    def _read_large_json(self, input_file):
        with open(input_file, 'rb') as f:
            return f

    # Read prompt templates from a CSV file
    def read_prompt_templates_from_csv(self, file_path: str):
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
        return prompt_templates

    def mask_entity(self, entity_tokens, tokenizer):
        tokens = tokenizer.tokenize(' ' + entity_tokens)
        to_mask = rnd.randint(0, len(tokens)-1)
        if type(tokenizer) == BertTokenizer:
            if '##' in tokens[to_mask]:
                bad_token_spl = True
                while bad_token_spl:
                    to_mask = rnd.randint(0, len(tokens) - 1)
                    if '##' not in tokens[to_mask]:
                        bad_token_spl = False
        masked_entity_tokens = tokens.copy()
        masked_entity_tokens[to_mask] = tokenizer.mask_token
        masked_entity = tokenizer.convert_tokens_to_string(masked_entity_tokens)
        entity_label = tokenizer.convert_tokens_to_string(tokens)
        return masked_entity, entity_label



    def create_prompt(self, splited_line: List, prompt_templates: dict,
                      tokenizer) -> str:
        relation = splited_line[0]
        subject_entity = splited_line[1].replace('_', ' ')
        obj_entities = splited_line[2].replace('_', ' ')

        prompt_template = prompt_templates[relation]
        prompt_template = prompt_template['LAMA']

        if rnd.random() < 0.5: # mask obj
            masked_obj, obj_label = self.mask_entity(obj_entities,tokenizer)
            if type(tokenizer) == BertTokenizer:
                prompt = prompt_template.replace('[X]', ' ' + subject_entity).replace('[Y]', masked_obj)
                label = prompt_template.replace('[X]', ' ' + subject_entity).replace('[Y]', obj_label)
            else:
                prompt = prompt_template.replace('[X]', ' ' + subject_entity).replace(' [Y]', masked_obj)
                label = prompt_template.replace('[X]', ' ' + subject_entity).replace(' [Y]', obj_label)
        else: # mask subj
            masked_subj, subj_label = self.mask_entity(subject_entity,tokenizer)
            prompt = prompt_template.replace('[X]', masked_subj).replace('[Y]', obj_entities)
            label = prompt_template.replace('[X]', subj_label).replace('[Y]', obj_entities)


        return prompt, label


    def _create_examples(self, input_file, set_type, debug=False, model_tokenizer=None, max_len=128, isTest=False, args=None):
        """Creates examples for the training and dev sets."""

        with open('./dat/qa_conceptNet/CN_CSQA/templates.json',
                  'r') as templates:
            templates_dict = json.load(templates)
        examples = []

        rnd.seed(args.data_seed)
        with open(input_file, 'r') as f:
            id_c = 0
            # data = json.load(f)

            for line in f:
                if id_c in [28, 40, 53, 73, 98, 207]:
                    print('Problem')
                    some_value = 0

                splited_line = line.strip().split('\t')

                text_a, label = self.create_prompt(splited_line=splited_line, prompt_templates=templates_dict,
                                                        tokenizer=model_tokenizer)

                if len(model_tokenizer.tokenize(text_a)) != len(model_tokenizer.tokenize(label)):
                    print('bad triple encoding: ', line)
                    print('id_c: ', id_c)
                    continue

                relation = splited_line[0]
                if debug and id_c > 10:
                    break
                tokens_in_triple = model_tokenizer.tokenize(text_a)
                if len(tokens_in_triple) > (max_len-4):
                    print('too long triple: ', text_a)
                    continue
                guid = "%s-%s" % (set_type, id_c)
                id_c += 1

                examples.append(
                    InputExample(guid=guid, text_a=text_a, label=label, relation_type=relation))
        print(len(examples))

        return examples





class AscProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_train_examples(self, data_dir, fn="train.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "train")

    def get_dev_examples(self, data_dir, fn="dev.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "dev")

    def get_test_examples(self, data_dir, fn="test.json"):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fn)), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, ids) in enumerate(lines):
            guid = "%s-%s" % (set_type, ids )
            text_a = lines[ids]['term']
            text_b = lines[ids]['sentence']
            label = lines[ids]['polarity']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples





class StringProcessor(DataProcessor):
    """Processor for the SemEval Aspect Sentiment Classification."""

    def get_examples(self, lines):
        """See base class."""
        return self._create_examples(lines)

    def _create_examples(self, lines):
        """Creates examples for the training and dev sets."""
        examples = []
        for line in lines:
            text_a = line
            examples.append(
                InputExample(text_a=text_a))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing

    if transformer_args.task in ['asc', 'asc_2', 'asc_5']: # for pair
        label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
    elif transformer_args.task == 'nli':
        label_map={'neutral': 0, 'entailment': 1, 'contradiction': 2}
    elif transformer_args.task == 'ae':
        label_map={'B': 0, 'I': 1, 'O': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        if mode!="ae":
            tokens_a = tokenizer.tokenize(example.text_a)
        else: #only do subword tokenization.
            tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # token_a has a max_length
        if transformer_args.exp in ['3layer_aspect','2layer_aspect_transfer','2layer_aspect_dynamic']:
            term_position = tokens.index('[SEP]')-1
            while term_position < transformer_args.max_term_length: #[CLS],t,[SEP]
                input_ids.insert(term_position,0)
                input_mask.insert(term_position,0)
                segment_ids.insert(term_position,0)
                term_position+=1

            max_seq_length = max_seq_length
        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if mode!="ae":
            label_id = label_map[example.label]
        else:
            label_id = [-1] * len(input_ids) #-1 is the index to ignore
            #truncate the label length if it exceeds the limit.
            lb=[label_map[label] for label in labels_a]
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features_w2v_dsc(examples, label_list,tokenizer,args):

    # prepare for word2vector experiments

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    label_map={'-1': 0, '1': 1}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_ids = tokenizer.texts_to_sequences([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        tokens_a_ids =  pad_sequences(tokens_a_ids, maxlen=args.max_seq_length,padding='post',value=0)[0]

        # print('example.text_a: ',example.text_a.translate(str.maketrans('', '', string.punctuation)).lower())
        # print('tokens_a_ids: ',tokens_a_ids)
        # print('tokens_a_ids: ',len(tokens_a_ids))

        # exit()
        if 'newsgroup' in args.task:
            label_id = example.label
        else:
            label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        tokens_term_ids=tokens_a_ids,
                        tokens_sentence_ids=tokens_a_ids,
                        label_id=label_id))
    return features

def convert_examples_to_features_w2v(examples, label_list,tokenizer,args):

    # prepare for word2vector experiments

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_ids = tokenizer.texts_to_sequences([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        tokens_b_ids = tokenizer.texts_to_sequences([example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])

        # if len(tokens_a_ids[0])<= 0 or len(tokens_b_ids[0])<= 0:
        #     print('empty')
        #     continue
        # assert len(tokens_a_ids[0]) > 0
        # assert len(tokens_b_ids[0]) > 0

        tokens_a_ids =  pad_sequences(tokens_a_ids, maxlen=args.max_term_length,padding='post',value=0)[0]
        tokens_b_ids =  pad_sequences(tokens_b_ids, maxlen=args.max_sentence_length,padding='post',value=0)[0]

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        tokens_term_ids=tokens_a_ids,
                        tokens_sentence_ids=tokens_b_ids,
                        label_id=label_id))

    return features


def convert_examples_to_features_w2v_as(examples, label_list,tokenizer,args):
    # w2v also considers aspect (by adding aspect at the beginning)
    # prepare for word2vector experiments

    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    # label_map = {}
    # for (i, label) in enumerate(label_list):
    #     label_map[label] = i

    if transformer_args.task == 'asc':
        label_map={'+': 0,'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
    elif transformer_args.task == 'nli':
        label_map={'neutral': 0, 'entailment': 1, 'contradiction': 2}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a_ids = tokenizer.texts_to_sequences([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        tokens_b_ids = tokenizer.texts_to_sequences([example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])

        tokens_ids = tokenizer.texts_to_sequences(
               [example.text_a.translate(str.maketrans('', '', string.punctuation)).lower() + ' ' +
               example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])

        # print([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower()])
        # print([example.text_a.translate(str.maketrans('', '', string.punctuation)).lower() + ' ' +
        #        example.text_b.translate(str.maketrans('', '', string.punctuation)).lower()])
        # print('tokens_ids: ',tokens_ids)
        # print('tokens_a_ids: ',tokens_a_ids)

        # if len(tokens_a_ids[0])<= 0 or len(tokens_b_ids[0])<= 0:
        #     print('empty')
        #     continue
        # assert len(tokens_a_ids[0]) > 0
        # assert len(tokens_b_ids[0]) > 0

        tokens_ids =  pad_sequences(tokens_ids, maxlen=args.max_seq_length,padding='post',value=0)[0]
        tokens_a_ids =  pad_sequences(tokens_a_ids, maxlen=args.max_term_length,padding='post',value=0)[0]
        tokens_b_ids =  pad_sequences(tokens_b_ids, maxlen=args.max_sentence_length,padding='post',value=0)[0]

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        tokens_term_sentence_ids=tokens_ids,
                        tokens_term_ids=tokens_a_ids,
                        tokens_sentence_ids=tokens_b_ids,
                        label_id=label_id))

    return features





def convert_examples_to_features_dtc(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    #TODO: input document only

    features = []
    for (ex_index,example) in enumerate(examples):
        labels_a = example.label
        tokens_a = tokenizer.tokenize(example.text_a)

        # print('labels_a: ',labels_a)
        # print('example.text_a: ',example.text_a)
        # print('tokens_a: ',tokens_a)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=labels_a))
    return features




def convert_examples_to_features_dsc(examples, label_list, max_seq_length, tokenizer,mode):
    """Loads a data file into a list of `InputBatch`s."""
    #TODO: input document only
    label_map={'-1': 0, '1': 1}
    if transformer_args.task == 'dsc':
        label_map={'-1': 0, '1': 1}
    elif transformer_args.task == 'ssc':
        label_map={'0': 0, '1': 1,'2': 2}

    features = []
    for (ex_index,example) in enumerate(examples):
        labels_a = label_map[example.label]
        tokens_a = tokenizer.tokenize(example.text_a)

        # print('labels_a: ',labels_a)
        # print('example.text_a: ',example.text_a)
        # print('tokens_a: ',tokens_a)


        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=labels_a))
    return features




def build_model_vocab_vector(tokenizer=None, tokens_list=None):
    model_vocab_dict = tokenizer.get_vocab()
    model_vocab_tokens = model_vocab_dict.keys()
    vocab_binary_vector = torch.zeros(tokenizer.vocab_size)
    common_words = []
    for token in tokens_list:
        token = tokenizer.tokenize(token)
        if len(token) == 1:
            vocab_binary_vector[model_vocab_dict[token[0]]] = 1
            # common_words.append(token)
    return vocab_binary_vector, tokens_list


def convert_examples_to_features_cn_bert(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    # TODO: input document only
    features = []
    for (ex_index, example) in enumerate(examples):

        labels_a = tokenizer.tokenize(example.label)
        tokens_a = tokenizer.tokenize(example.text_a)

        # print('labels_a: ',labels_a)
        # print('tokens_a: ',tokens_a)


        assert len(tokens_a) == len(labels_a), f"wrong token length: {tokens_a}, {labels_a}"

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            # tokens_a = tokens_a[0:(max_seq_length - 2)]
            tokens_a = tokens_a[len(tokens_a)-(max_seq_length-2):]

        if len(labels_a) > max_seq_length - 2:
            # labels_a = labels_a[0:(max_seq_length - 2)]
            labels_a = labels_a[len(labels_a)-(max_seq_length-2):]

        tokens = []
        label_tokens = []
        tokens.append(tokenizer.cls_token)
        label_tokens.append(tokenizer.cls_token)
        for token in tokens_a:
            tokens.append(token)
        tokens.append(tokenizer.sep_token)
        for label_token in labels_a:
            label_tokens.append(label_token)
        label_tokens.append(tokenizer.sep_token)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = tokenizer.convert_tokens_to_ids(label_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * max_seq_length

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.pad_token_id)
            input_mask.append(0)

        while len(label_ids) < max_seq_length:
            label_ids.append(tokenizer.pad_token_id)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        # labels = [-100 if x == tokenizer.pad_token_id else x for x in label_ids]
        labels = [-100 if x != tokenizer.mask_token_id else label_ids[t_pos] for t_pos, x in enumerate(input_ids)]

        # spec_tokens = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
        # labels = [-100 if x in spec_tokens else x for x in label_ids]

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=labels))
    return features



def convert_examples_to_features_csqa_bert(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    num_croped_questions = 0

    for (ex_index, example) in enumerate(examples):

        # label = len(label_map) * [0]
        # label[label_map[example.label]] = 1
        label = label_map[example.label]

        question_tokens = tokenizer.tokenize(example.text_a)



        # Account for <eos> and 2x <SEP> with "- 3"
        if example.max_pair_len > max_seq_length - 4:
            question_tokens = question_tokens[0:(max_seq_length - 4)]
            num_croped_questions += 1

        # concatenate question and answer choices
        question_answer_pairs_token_ids = []
        input_masks = []
        segment_ids = []
        for ch in example.choices:
            choice_tokens = tokenizer.tokenize(ch)
            qa_tokens = [tokenizer.cls_token] + question_tokens + [tokenizer.sep_token] + [tokenizer.sep_token] + choice_tokens + [tokenizer.sep_token]
            qa_ids = tokenizer.convert_tokens_to_ids(qa_tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(qa_ids)
            segment_id = [0] * len(qa_ids)

            # Zero-pad up to the sequence length.
            while len(qa_ids) < max_seq_length:
                qa_ids.append(tokenizer.pad_token_id)
                input_mask.append(0)
                segment_id.append(0)

            assert len(qa_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_id) == max_seq_length

            question_answer_pairs_token_ids.append(qa_ids)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)

        features.append(
                InputFeatures(
                        input_ids=question_answer_pairs_token_ids,
                        input_mask=input_masks,
                        segment_ids=segment_ids,
                        label_id=label))
    return features




def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens



def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


