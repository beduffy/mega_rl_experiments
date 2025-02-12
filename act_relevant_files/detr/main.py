# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    # parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    # parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    # parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    # parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')




    # new defaults to prevent error running act from ACT
    parser.add_argument('--seed', default=42, action='store', type=int, help='seed')
    parser.add_argument('--num_epochs', default=2000, action='store', type=int, help='num_epochs')
    parser.add_argument('--ckpt_dir', default='checkpoints', action='store', type=str, help='ckpt_dir')
    parser.add_argument('--policy_class', default='ACT', action='store', type=str, help='policy_class, capitalize')


    # Device arguments
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        type=str, choices=['cpu', 'cuda'],
                        help="Device to use for training/evaluation")


    return parser


def build_ACT_model_and_optimizer(args_override):
    """Build the ACT model and optimizer
    
    Args:
        args_override: Dictionary of arguments to override defaults
    """
    # Create default args
    args = argparse.Namespace()
    
    # Set default values based on typical usage
    args.task_name = 'sim_transfer_cube_scripted'
    args.ckpt_dir = 'checkpoints'
    args.policy_class = 'ACT'
    args.kl_weight = 10
    args.chunk_size = 100
    args.hidden_dim = 512
    args.batch_size = 1
    args.dim_feedforward = 3200
    args.num_epochs = 2000
    args.lr = 1e-5
    args.seed = 0
    args.device = 'cpu'
    
    # Additional required args for the model
    args.num_queries = 3
    args.enc_layers = 2
    args.dec_layers = 2
    args.nheads = 8
    args.dropout = 0.1
    args.backbone = 'resnet18'
    args.position_embedding = 'sine'
    args.lr_backbone = 1e-5
    args.weight_decay = 1e-4
    args.camera_names = ['dummy']
    args.state_dim = 1
    args.num_actions = 1
    args.masks = False
    args.dilation = False  # Default value for dilation
    args.pre_norm = False  # Default value for pre_norm
    
    # Override with any provided arguments
    if isinstance(args_override, dict):
        for key, value in args_override.items():
            setattr(args, key, value)
    
    # Create model
    model = build_ACT_model(args)
    
    # Create optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    
    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.to(args.device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

