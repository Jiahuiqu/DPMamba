# --------------------------------------------------------
# Modified by Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import time
import datetime

import yaml
from yacs.config import CfgNode as CN
import argparse

_C = CN()

# Base config files
_C.BASE = ['']
# ===
_C.MASTER_PORT_ = 29502 # 显示占用的端口
_C.CUDA_VISIBLE_DEVICES_ = 0 # cuda:0
# ===
# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()

# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 64
# Modality_Num
_C.DATA.Modality_Num = 2
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET_NAME = 'MUUFL'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6

_C.DATA.PATCH_SIZE = 9
_C.DATA.Num_Samples_Per_Class = 40
# Number of classes,
_C.DATA.NUM_CLASSES = 11
_C.DATA.N_PCA = 32

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'DPMamba'
# Model name
_C.MODEL.NAME = 'baseline'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

#
_C.MODEL.HIDDEN_DIM = 64
_C.MODEL.HIDDEN_DIM_IMAM = 64
_C.MODEL.HIDDEN_DIM_CMAM = 128
_C.MODEL.SHARE_SSM = False

# MMpretrain models for test 11
_C.MODEL.MMCKPT = False

# VSSM parameters
_C.MODEL.VSSM = CN()
_C.MODEL.VSSM.SCAN_MODE = "Zigzag"
_C.MODEL.VSSM.PATCH_SIZE = 4
_C.MODEL.VSSM.IN_CHANS = 3
_C.MODEL.VSSM.DEPTHS = [2, 2, 9, 2]
_C.MODEL.VSSM.EMBED_DIM = [[64, 32], [32, 64]]
_C.MODEL.VSSM.SSM_D_STATE = 16
_C.MODEL.VSSM.SSM_RATIO = 2.0
_C.MODEL.VSSM.SSM_RANK_RATIO = 2.0
_C.MODEL.VSSM.SSM_DT_RANK = "auto"
_C.MODEL.VSSM.SSM_ACT_LAYER = "silu"
_C.MODEL.VSSM.SSM_CONV = 3
_C.MODEL.VSSM.SSM_CONV_BIAS = True
_C.MODEL.VSSM.SSM_DROP_RATE = 0.0
_C.MODEL.VSSM.SSM_INIT = "v0"
_C.MODEL.VSSM.SSM_FORWARDTYPE = "v2"
_C.MODEL.VSSM.MLP_RATIO = 4.0
_C.MODEL.VSSM.MLP_ACT_LAYER = "gelu"
_C.MODEL.VSSM.MLP_DROP_RATE = 0.0
_C.MODEL.VSSM.PATCH_NORM = True
_C.MODEL.VSSM.NORM_LAYER = "ln"
_C.MODEL.VSSM.DOWNSAMPLE = "v2"
_C.MODEL.VSSM.PATCHEMBED = "v2"
_C.MODEL.VSSM.POSEMBED = False
_C.MODEL.VSSM.GMLP = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300 # ！！！！！！！！！！！！！！！！！！！！！！！！！！！
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.LEARN_RATE = 5e-4 #
_C.TRAIN.BASE_LR = 5e-4 #
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
_C.TRAIN.EPOCH_CHANGE_OPTI = 100

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []
# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
_C.Is_Pretrain = False
_C.MODEL_PATH = ""
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Test traincost only, overwritten by command line argument
_C.TRAINCOST_MODE = False
# for acceleration
_C.FUSED_LAYERNORM = False

# ==============================================================
#  Fine tuning setting
# ==============================================================
_C.Fine_TUNE = CN()
_C.Fine_TUNE.START_EPOCH = 0
_C.Fine_TUNE.EPOCHS = 300 #
_C.Fine_TUNE.WARMUP_EPOCHS = 20
_C.Fine_TUNE.WEIGHT_DECAY = 0.05
_C.Fine_TUNE.LEARN_RATE = 5e-4 #
_C.Fine_TUNE.BASE_LR = 5e-4 #
_C.Fine_TUNE.WARMUP_LR = 5e-7
_C.Fine_TUNE.MIN_LR = 5e-6
# Clip gradient norm
_C.Fine_TUNE.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.Fine_TUNE.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.Fine_TUNE.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.Fine_TUNE.USE_CHECKPOINT = False
_C.Fine_TUNE.EPOCH_CHANGE_OPTI = 100

# LR scheduler
_C.Fine_TUNE.LR_SCHEDULER = CN()
_C.Fine_TUNE.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.Fine_TUNE.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.Fine_TUNE.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.Fine_TUNE.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.Fine_TUNE.LR_SCHEDULER.GAMMA = 0.1
_C.Fine_TUNE.LR_SCHEDULER.MULTISTEPS = []
# Optimizer
_C.Fine_TUNE.OPTIMIZER = CN()
_C.Fine_TUNE.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.Fine_TUNE.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.Fine_TUNE.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.Fine_TUNE.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.Fine_TUNE.LAYER_DECAY = 1.0

# MoE
_C.Fine_TUNE.MOE = CN()
# Only save model on master device
_C.Fine_TUNE.MOE.SAVE_MASTER = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batchsize'):
        config.DATA.BATCH_SIZE = args.batchsize
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True
    if _check_args('traincost'):
        config.TRAINCOST_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    ##
    parser.add_argument('--cfg', type=str, metavar="FILE", default="./configs/baseline.yaml", help='path to config file', )
    parser.add_argument('--Is_Pretrain', type=bool, default=True, help='Is Pretrain')
    parser.add_argument('--MODEL_PATH', type=str, default="", help='Model Path')
    ##
    parser.add_argument('--output', type=str, metavar='PATH', default='./output',  ### 修改此处
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    # parser.add_argument('--batchsize', type=int, default=64)
    # parser.add_argument('--Modality_Num', type=int, default=2)
    # parser.add_argument('--share_SSM', type=bool, default=False)
    # parser.add_argument('--patchsize', type=int, default=9)
    # parser.add_argument('--hidden_dim', type=int, default=96)
    # parser.add_argument('--dataset_name', type=str, default="MUUFL")
    # parser.add_argument('--num_samples_per_class', type=int, default=40)
    # parser.add_argument('--class_num', type=int, default=11)
    # parser.add_argument('--learning_rate', type=float, default=5e-4)

    # =========================================================================================================
    parser.add_argument('--data-path', type=str, default=" ", help='path to dataset')  ##
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')

    parser.add_argument('--tag', default=time.strftime("%Y%m%d%H%M%S", time.localtime()), help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    parser.add_argument('--optim', type=str, help='overwrite optimizer if provided, can be adamw/sgd.')

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=True)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')

    parser.add_argument('--memory_limit_rate', type=float, default=-1, help='limitation of gpu memory use')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config