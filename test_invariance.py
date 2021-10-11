import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn.datasets as datasets
import flows.layers as layers
import wandb
import math
import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
import pprint
from functools import partial
import ipdb
import gc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
from utils import utils
from utils.utils import seed_everything, str2bool, visualize_transform
from data import create_dataset
from data.toy_data import sample_2d_data
from flows import create_flow
from e2cnn import gspaces
from e2cnn import nn as enn
import warnings
warnings.filterwarnings("ignore")



def train_flow(args, flow_model, optim, scheduler, train_loader, test_loader):
    flow_model.eval()
    last_checkpoints = None
    for i, (x, y) in enumerate(tqdm(test_loader)):
        x = x.to(args.dev)
        ipdb.set_trace()
        flow_model.check_invariant_log_prob(flow_model.group_action_type,
                                            flow_model.input_type, x)


def main(args):
    train_loader, test_loader = create_dataset(args, args.dataset)
    if args.task in ['classification', 'hybrid']:
        try:
            args.n_classes
        except NameError:
            raise ValueError('Cannot perform classification with {}'.format(args.dataset))
    else:
        args.n_classes = 1
    args.input_size = (args.batch_size, args.im_dim + args.padding, args.imagesize, args.imagesize)

    if args.squeeze_first:
        args.input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    if args.model_type == 'E_resflow' or args.model_type == 'Mixed_resflow':
        # args.init_layer = layers.EquivariantLogitTransform(args.logit_init)
        args.init_layer = None
        args.squeeze_layer = layers.EquivariantSqueezeLayer(2)
    else:
        args.init_layer = layers.LogitTransform(args.logit_init)
        args.squeeze_layer = layers.SqueezeLayer(2)

    flow = create_flow(args, args.model_type)
    print("Number of trainable parameters: {}".format(utils.count_parameters(flow)))

    scheduler = None
    flow_params = filter(lambda p: p.requires_grad, flow.parameters())
    if args.optimizer == 'adam':
        optimizer = optim.Adam(flow_params, lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
        if args.scheduler: scheduler = utils.CosineAnnealingWarmRestarts(optimizer, 20, T_mult=2, last_epoch=args.begin_epoch - 1)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(flow.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(flow.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(flow.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[60, 120, 160], gamma=0.2, last_epoch=args.begin_epoch - 1
            )
    else:
        raise ValueError('Unknown optimizer {}'.format(args.optimizer))
    utils.load_checkpoint(args, args.resume_path, flow, optimizer)
    train_flow(args, flow, optimizer, scheduler, train_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # boiler plate inits
    parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
    parser.add_argument('--cuda', type=int, help='Which GPU to run on.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--plot_name', type=str, default='test')
    # target density
    parser.add_argument('--model_type', type=str, default="Toy", help='Which Flow to use.')
    parser.add_argument('--dataset', type=str, default=None, help='Which potential function to approximate.')
    # model parameters
    parser.add_argument('--imagesize', type=int, default=28)
    # i-Resnet params
    parser.add_argument('--coeff', type=float, default=0.98)
    parser.add_argument('--logit_init', type=float, default=1e-6, help='Logit layer init')
    parser.add_argument('--vnorms', type=str, default='2222')
    parser.add_argument('--n-lipschitz-iters', type=int, default=200)
    parser.add_argument('--sn-tol', type=float, default=1e-3)
    parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)

    parser.add_argument('--act', type=str, default='swish')
    parser.add_argument('--idim', type=int, default=512)
    parser.add_argument('--n_blocks', type=str, default='16-16-16')
    parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
    parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
    parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
    parser.add_argument('--kernels', type=str, default='3-1-3')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--add-noise', type=eval, choices=[True, False], default=True)
    parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
    parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
    parser.add_argument('--fc-idim', type=int, default=128)
    parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--realnvp-padding', type=int, default=1)
    parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
    parser.add_argument('--cdim', type=int, default=256)
    parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

    parser.add_argument('--dims', type=str, default='128-128-128-128')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of hidden layers.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Dimensions of hidden layers.')
    parser.add_argument('--brute-force', type=eval, choices=[True, False], default=False)
    parser.add_argument('--exact-trace', type=eval, choices=[True, False], default=False)
    parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)
    parser.add_argument('--n-power-series', type=int, default=None)
    parser.add_argument('--n-exact-terms', type=int, default=10)
    parser.add_argument('--var-reduc-lr', type=float, default=0)
    parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
    parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)
    parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')
    parser.add_argument('--out-fiber', type=str, default='regular')
    parser.add_argument('--field-type', type=int, default=0, help='Only For Continuous groups. Picks the frequency.')
    parser.add_argument('--n-samples', type=int, default=10)
    parser.add_argument('--group', type=str, default='fliprot4', help='The choice of group representation for Equivariance')
    # training parameters
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
    parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
    parser.add_argument('--num_iters', type=int, default=10000)
    parser.add_argument('--batch_size', help='Minibatch size', type=int, default=64)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--annealing-iters', type=int, default=0)
    parser.add_argument('--save', help='directory to save results', type=str, default='runs/')
    parser.add_argument('--resume_path', help='directory to save results', type=str, default='runs/')
    parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
    parser.add_argument('--validation', type=eval, choices=[True, False], default=True)
    parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
    parser.add_argument('--update-freq', type=int, default=1)

    parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
    parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
    parser.add_argument('--update-lr', type=eval, choices=[True, False], default=False)
    parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
    parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')
    parser.add_argument('--double-padding', type=eval, choices=[True, False], default=False)

    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--begin-epoch', type=int, default=0)

    parser.add_argument('--augment', dest="augment", action="store_true",
                        help='Augment the training set with rotated images')
    parser.add_argument('--interpolation', type=int, default=2,
                        help='Type of interpolation to use for data augmentation')

    parser.add_argument('--nworkers', type=int, default=4)
    parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
    parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=500)
    parser.add_argument("--wandb", action="store_true", default=False, help='Use wandb for logging')
    parser.add_argument('--namestr', type=str, default='E-Resflow', \
            help='additional info in output filename to describe experiments')
    args = parser.parse_args()

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file
    if os.path.isfile("settings.json"):
        with open('settings.json') as f:
            data = json.load(f)
        args.wandb_apikey = data.get("wandbapikey")

    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_apikey
        wandb.init(project='Equivariant-Flows',
                   name='Equivariant-Flows-{}-{}'.format(args.dataset, args.namestr))

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name = utils.project_name(args.dataset)


    main(args)
