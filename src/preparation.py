import torch
import os
import numpy as np
import logging
import sys
import random
from config import set_args
from pathlib import Path
from transformers import RobertaConfig

args = set_args()

# if you want to use the reset parameters --------------------------
if args.use_predefine_args:
    import load_base_args

    args = load_base_args.load()

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#
if not args.multi_gpu: torch.autograd.set_detect_anomaly(True)

# ----------------------------------------------------------------------
# Create needed folder, results name, reult matrix, if any.
# ----------------------------------------------------------------------

if args.output == '':
    args.output = './res/' + args.scenario + '/' + args.task + '/' + args.yaml_param_num + '/' + args.backbone + '_' + args.baseline + '_' + str(
        args.note) + '.txt'

model_path = './models/' + args.scenario + '/' + args.task + '/' + args.yaml_param_num + '/'
graph_model_path = './models/' + args.scenario + '/' + args.task + '/graph_run_' + args.yaml_param_num + '_' \
                   + args.graph_run + '_m_eps' + str(args.num_train_epochs) + '_' + str(args.seed) +'/'
res_path = './res/' + args.scenario + '/' + args.task + '/' + args.yaml_param_num + '/'
if args.model_save_path == '':
    args.model_save_path = model_path

if args.model_path == '':
    args.model_path = model_path

if not os.path.isdir(res_path): os.makedirs(res_path)
if not os.path.isdir(model_path): os.makedirs(model_path)
if args.graph_run != '' and not os.path.isdir(graph_model_path): os.makedirs(graph_model_path)

performance_output = args.output + '_performance'
performance_output_forward = args.output + '_forward_performance'
f1_macro_output = args.output + '_f1_macro'
f1_macro_output_forward = args.output + '_forward_f1_macro'

precision_avg_output = args.output + '_precision_avg'
precision_avg_output_forward = args.output + '_forward_precision_avg'
recall_avg_output = args.output + '_recall_avg'
recall_avg_output_forward = args.output + '_forward_recall_avg'
f1_avg_output = args.output + '_f1_avg'
f1_avg_output_forward = args.output + '_forward_f1_avg'

performance_d_prec = args.output + '_performance_d_prec'
performance_d_recall = args.output + '_performance_d_reacll'
performance_d_f1 = args.output + '_performance_d_f1'

acc = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
lss = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
f1_macro = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

precision_avg = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
recall_avg = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)
f1_avg = np.zeros((args.ntasks, args.ntasks), dtype=np.float32)

base_model_path = args.model_path
base_resume_from_file = args.resume_from_file

#
# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)  # for random sample
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')  # sys.exit() # for working locally on cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

classification_tasks = ['asc', 'dsc', 'ssc', 'newsgroup',
                        'celeba', 'femnist', 'cifar10', 'mnist', 'fashionmnist', 'cifar100', 'vlcs', 'asc_2', 'asc_5',
                        'qa_dragon']
extraction_tasks = ['ner', 'ae']


def resume_checkpoint(appr, net):
    if args.auto_resume:
        auto_resume_dir = Path(args.model_path)
        if auto_resume_dir.is_dir():
            save_files = list(auto_resume_dir.glob('steps*'))
            if len(save_files) > 0:
                last_task_id = max([int(path.name.rstrip('_mask_pre').rstrip('_mask_bac').lstrip('steps')) for path in save_files])
                args.resume_model = True
                args.resume_from_file = str(auto_resume_dir / f'steps{last_task_id}')
                args.resume_from_task = last_task_id + 1

    if args.resume_model:
        if torch.cuda.is_available(): # GPU available
            checkpoint = torch.load(args.resume_from_file)
        else:
            checkpoint = torch.load(args.resume_from_file, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model_state_dict'])
        logger.info('resume_from_file: ' + str(args.resume_from_file))

    if args.resume_model:
        if hasattr(appr, 'mask_pre'):
            if torch.cuda.is_available():  # GPU available
                appr.mask_pre = torch.load(args.resume_from_file + '_mask_pre')  # not in state_dict
            else:
                appr.mask_pre = torch.load(args.resume_from_file + '_mask_pre', map_location=torch.device('cpu'))  # not in state_dict
        if hasattr(appr, 'mask_back'):
            if torch.cuda.is_available():  # GPU available
                appr.mask_back = torch.load(args.resume_from_file + '_mask_back')
            else:
                appr.mask_back = torch.load(args.resume_from_file + '_mask_back', map_location=torch.device('cpu'))

        # for GEM
        if hasattr(appr, 'buffer'):
            appr.buffer = torch.load(args.resume_from_file + '_buffer')  # not in state_dict
        if hasattr(appr, 'grad_dims'):
            appr.grad_dims = torch.load(args.resume_from_file + '_grad_dims')  # not in state_dict
        if hasattr(appr, 'grads_cs'):
            appr.grads_cs = torch.load(args.resume_from_file + '_grads_cs')  # not in state_dict
        if hasattr(appr, 'grads_da'):
            appr.grads_da = torch.load(args.resume_from_file + '_grads_da')  # not in state_dict
        if hasattr(appr, 'history_mask_pre'):
            appr.history_mask_pre = torch.load(args.resume_from_file + '_history_mask_pre')  # not in state_dict
        if hasattr(appr, 'similarities'):
            appr.similarities = torch.load(args.resume_from_file + '_similarities')  # not in state_dict
        if hasattr(appr, 'check_federated'):
            appr.check_federated = torch.load(args.resume_from_file + '_check_federated')  # not in state_dict


def resume_checkpoint_with_graph(appr, net):
    if args.auto_resume:
        auto_resume_dir = Path(args.model_path)
        if auto_resume_dir.is_dir():
            save_files = list(auto_resume_dir.glob('steps*'))
            if len(save_files) > 0:
                last_task_id = 0#max([int(path.name.rstrip('_mask_pre').rstrip('_mask_bac').lstrip('steps')) for path in save_files])
                args.resume_model = True
                args.resume_from_file = str(auto_resume_dir / f'steps{0}')
                if args.graph_run == 'last':
                    args.resume_from_file = str(auto_resume_dir / f'steps_last{0}')
                args.resume_from_task = last_task_id + 1

    if args.resume_model:
        if torch.cuda.is_available(): # GPU available
            checkpoint = torch.load(args.resume_from_file)
        else:
            checkpoint = torch.load(args.resume_from_file, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model_state_dict'])
        logger.info('resume_from_file: ' + str(args.resume_from_file))

    if args.resume_model:
        if hasattr(appr, 'mask_pre'):
            if torch.cuda.is_available():  # GPU available
                appr.mask_pre = torch.load(args.resume_from_file + '_mask_pre')  # not in state_dict
            else:
                appr.mask_pre = torch.load(args.resume_from_file + '_mask_pre', map_location=torch.device('cpu'))  # not in state_dict
        if hasattr(appr, 'mask_back'):
            if torch.cuda.is_available():  # GPU available
                appr.mask_back = torch.load(args.resume_from_file + '_mask_back')
            else:
                appr.mask_back = torch.load(args.resume_from_file + '_mask_back', map_location=torch.device('cpu'))

        # for GEM
        if hasattr(appr, 'buffer'):
            appr.buffer = torch.load(args.resume_from_file + '_buffer')  # not in state_dict
        if hasattr(appr, 'grad_dims'):
            appr.grad_dims = torch.load(args.resume_from_file + '_grad_dims')  # not in state_dict
        if hasattr(appr, 'grads_cs'):
            appr.grads_cs = torch.load(args.resume_from_file + '_grads_cs')  # not in state_dict
        if hasattr(appr, 'grads_da'):
            appr.grads_da = torch.load(args.resume_from_file + '_grads_da')  # not in state_dict
        if hasattr(appr, 'history_mask_pre'):
            appr.history_mask_pre = torch.load(args.resume_from_file + '_history_mask_pre')  # not in state_dict
        if hasattr(appr, 'similarities'):
            appr.similarities = torch.load(args.resume_from_file + '_similarities')  # not in state_dict
        if hasattr(appr, 'check_federated'):
            appr.check_federated = torch.load(args.resume_from_file + '_check_federated')  # not in state_dict
    if args.resume_model:
        args.model_save_path = graph_model_path
        args.model_path = graph_model_path
        global base_model_path
        base_model_path = graph_model_path

    if args.train_base_model:
        print('Train K-ADAPTER: Base model unfreezed')
        config = RobertaConfig.from_pretrained(args.bert_model)
        for param in net.roberta.parameters():
            param.requires_grad = True

        # Only adapters are trainable

        if args.apply_roberta_output and args.apply_roberta_attention_output:
            adaters = \
                [net.roberta.encoder.layer[layer_id].attention.output.adapter for layer_id in
                 range(config.num_hidden_layers)] + \
                [net.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in
                 range(config.num_hidden_layers)] + \
                [net.roberta.encoder.layer[layer_id].output.adapter for layer_id in
                 range(config.num_hidden_layers)] + \
                [net.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in
                 range(config.num_hidden_layers)]

        elif args.apply_roberta_output:
            adaters = \
                [net.roberta.encoder.layer[layer_id].output.adapter for layer_id in
                 range(config.num_hidden_layers)] + \
                [net.roberta.encoder.layer[layer_id].output.LayerNorm for layer_id in
                 range(config.num_hidden_layers)]

        elif net.apply_roberta_attention_output:
            adaters = \
                [net.roberta.encoder.layer[layer_id].attention.output.adapter for layer_id in
                 range(config.num_hidden_layers)] + \
                [net.roberta.encoder.layer[layer_id].attention.output.LayerNorm for layer_id in
                 range(config.num_hidden_layers)]

        for adapter in adaters:
            for param in adapter.parameters():
                param.requires_grad = False



def resume_checkpoint_for_eval(appr, net, step):
    auto_resume_dir = Path(args.model_path)
    if args.auto_resume:
        args.resume_model = True
        args.resume_from_file = str(auto_resume_dir / f'steps{step}')
        args.resume_from_task = step + 1

    if args.resume_model:
        checkpoint = torch.load(args.resume_from_file)
        net.load_state_dict(checkpoint['model_state_dict'])
        logger.info('resume_from_file: ' + str(args.resume_from_file))

    if args.resume_model:
        if hasattr(appr, 'mask_pre'):
            appr.mask_pre = torch.load(args.resume_from_file + '_mask_pre')  # not in state_dict
        if hasattr(appr, 'mask_back'):
            appr.mask_back = torch.load(args.resume_from_file + '_mask_back')

        # for GEM
        if hasattr(appr, 'buffer'):
            appr.buffer = torch.load(args.resume_from_file + '_buffer')  # not in state_dict
        if hasattr(appr, 'grad_dims'):
            appr.grad_dims = torch.load(args.resume_from_file + '_grad_dims')  # not in state_dict
        if hasattr(appr, 'grads_cs'):
            appr.grads_cs = torch.load(args.resume_from_file + '_grads_cs')  # not in state_dict
        if hasattr(appr, 'grads_da'):
            appr.grads_da = torch.load(args.resume_from_file + '_grads_da')  # not in state_dict
        if hasattr(appr, 'history_mask_pre'):
            appr.history_mask_pre = torch.load(args.resume_from_file + '_history_mask_pre')  # not in state_dict
        if hasattr(appr, 'similarities'):
            appr.similarities = torch.load(args.resume_from_file + '_similarities')  # not in state_dict
        if hasattr(appr, 'check_federated'):
            appr.check_federated = torch.load(args.resume_from_file + '_check_federated')  # not in state_dict

