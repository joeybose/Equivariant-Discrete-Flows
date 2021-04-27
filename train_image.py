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

criterion = torch.nn.CrossEntropyLoss()

def do_train_epoch(args, epoch, flow_model, optim, train_loader, meters):
    flow_model.train()
    total = 0
    correct = 0
    end = time.time()
    batch_time, bpd_meter, logpz_meter, deltalogp_meter, firmom_meter, secmom_meter, gnorm_meter, ce_meter, ema = meters[:]
    for i, (x, y) in enumerate(train_loader):
        global_itr = epoch * len(train_loader) + i
        utils.update_lr(args, optim, global_itr)

        # Training procedure:
        # for each sample x:
        #   compute z = f(x)
        #   maximize log p(x) = log p(z) - log |det df/dx|

        x = x.to(args.dev)

        beta = beta = min(1, global_itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        bpd, logits, logpz, neg_delta_logp = flow_model.compute_loss(args, x, beta=beta)

        if args.task in ['density', 'hybrid']:
            firmom, secmom = utils.estimator_moments(flow_model)

            bpd_meter.update(bpd.item())
            logpz_meter.update(logpz.item())
            deltalogp_meter.update(neg_delta_logp.item())
            firmom_meter.update(firmom)
            secmom_meter.update(secmom)

        if args.task in ['classification', 'hybrid']:
            y = y.to(args.dev)
            crossent = criterion(logits, y)
            ce_meter.update(crossent.item())

            # Compute accuracy.
            _, predicted = logits.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        # compute gradient and do SGD step
        if args.task == 'density':
            loss = bpd
        elif args.task == 'classification':
            loss = crossent
        else:
            if not args.scale_dim: bpd = bpd * (args.imagesize * args.imagesize * im_dim)
            loss = bpd + crossent / np.log(2)  # Change cross entropy from nats to bits.
        loss.backward()

        # if global_itr % args.update_freq == args.update_freq - 1:

            # if args.update_freq > 1:
                # with torch.no_grad():
                    # for p in flow_model.parameters():
                        # if p.grad is not None:
                            # p.grad /= args.update_freq

        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(flow_model.parameters(), 1.)
        if args.learn_p: flow_model.compute_p_grads(model)

        optim.step()
        optim.zero_grad()
        if 'resflow' in args.model_type:
            flow_model.update_lipschitz()
        ema.apply()

        gnorm_meter.update(grad_norm)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            s = (
                'Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} | '
                'GradNorm {gnorm_meter.avg:.2f}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, gnorm_meter=gnorm_meter
                )
            )

            if args.task in ['density', 'hybrid']:
                s += (
                    ' | Bits/dim {bpd_meter.val:.4f}({bpd_meter.avg:.4f}) | '
                    'Logpz {logpz_meter.avg:.0f} | '
                    '-DeltaLogp {deltalogp_meter.avg:.0f} | '
                    'EstMoment ({firmom_meter.avg:.0f},{secmom_meter.avg:.0f})'.format(
                        bpd_meter=bpd_meter, logpz_meter=logpz_meter, deltalogp_meter=deltalogp_meter,
                        firmom_meter=firmom_meter, secmom_meter=secmom_meter
                    )
                )

                if args.wandb:
                    wandb.log({'BPD': bpd_meter.val, "Logpz": logpz_meter.avg,
                               '-DeltaLogp': deltalogp_meter.avg, 'GradNorm':
                               gnorm_meter.avg, 'epoch': epoch})

            if args.task in ['classification', 'hybrid']:
                s += ' | CE {ce_meter.avg:.4f} | Acc {0:.4f}'.format(100 * correct / total, ce_meter=ce_meter)

            print(s)
        # if i % args.vis_freq == 0:
            # utils.visualize(args, epoch, flow_model, i, x)

        del x
        torch.cuda.empty_cache()
        gc.collect()
    pass

def validate(args, epoch, flow_model, test_loader, ema=None):
    """
    Evaluates the cross entropy between p_data and p_model.
    """
    bpd_meter = utils.AverageMeter()
    ce_meter = utils.AverageMeter()

    if ema is not None:
        ema.swap()

    if 'resflow' in args.model_type:
        flow_model.update_lipschitz()

    # flow_model = utils.parallelize(flow_model)
    flow_model.eval()

    correct = 0
    total = 0

    start = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(test_loader)):
            x = x.to(args.dev)
            bpd, logits, _, _ = flow_model.compute_loss(args, x)
            bpd_meter.update(bpd.item(), x.size(0))

            if args.task in ['classification', 'hybrid']:
                y = y.to(args.dev)
                loss = criterion(logits, y)
                ce_meter.update(loss.item(), x.size(0))
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
    val_time = time.time() - start

    if ema is not None:
        ema.swap()
    s = 'Epoch: [{0}]\tTime {1:.2f} | Test bits/dim {bpd_meter.avg:.4f}'.format(epoch, val_time, bpd_meter=bpd_meter)

    if args.wandb:
        wandb.log({'Test BPD': bpd_meter.avg})

    if args.task in ['classification', 'hybrid']:
        s += ' | CE {:.4f} | Acc {:.2f}'.format(ce_meter.avg, 100 * correct / total)
    print(s)
    return bpd_meter.avg


def train_flow(args, flow, optim, scheduler, train_loader, test_loader):
    batch_time = utils.RunningAverageMeter(0.97)
    bpd_meter = utils.RunningAverageMeter(0.97)
    logpz_meter = utils.RunningAverageMeter(0.97)
    deltalogp_meter = utils.RunningAverageMeter(0.97)
    firmom_meter = utils.RunningAverageMeter(0.97)
    secmom_meter = utils.RunningAverageMeter(0.97)
    gnorm_meter = utils.RunningAverageMeter(0.97)
    ce_meter = utils.RunningAverageMeter(0.97)
    ema = utils.ExponentialMovingAverage(flow)

    meters = [batch_time, bpd_meter, logpz_meter, deltalogp_meter,
              firmom_meter, secmom_meter, gnorm_meter, ce_meter, ema]
    lipschitz_constants = []

    for epoch in range(args.num_iters):

        print('Current LR {}'.format(optim.param_groups[0]['lr']))

        do_train_epoch(args, epoch, flow, optim, train_loader, meters)
        if 'resflow' in args.model_type:
            lipschitz_constants.append(flow.get_lipschitz_constants())
            print('Lipsch: {}'.format(utils.pretty_repr(lipschitz_constants[-1])))

        if args.ema_val:
            test_bpd = validate(args, epoch, flow, test_loader, ema)
        else:
            test_bpd = validate(args, epoch, flow, test_loader)

        if args.scheduler and scheduler is not None:
            scheduler.step()

    test_bpd = validate(args, epoch, flow, test_loader)

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
    if args.model_type == 'E_resflow':
        args.init_layer = layers.EquivariantLogitTransform(args.logit_init)
        args.squeeze_layer = layers.EquivariantSqueezeLayer(2)
    else:
        args.init_layer = layers.LogitTransform(args.logit_init)
        args.squeeze_layer = layers.SqueezeLayer(2)

    flow = create_flow(args, args.model_type)
    print("Number of trainable parameters: {}".format(utils.count_parameters(flow)))

    scheduler = None

    if args.optimizer == 'adam':
        optimizer = optim.Adam(flow.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.wd)
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
    parser.add_argument('--n-lipschitz-iters', type=int, default=None)
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
    parser.add_argument('--n-exact-terms', type=int, default=2)
    parser.add_argument('--var-reduc-lr', type=float, default=0)
    parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
    parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)
    parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')
    parser.add_argument('--out-fiber', type=str, default='regular')
    parser.add_argument('--field-type', type=int, default=0, help='Only For Continuous groups. Picks the frequency.')
    parser.add_argument('--n-samples', type=int, default=1)
    parser.add_argument('--group', type=str, default='fliprot4', help='The choice of group representation for Equivariance')
    # training parameters
    parser.add_argument('--optimizer', type=str, choices=['adam', 'adamax', 'rmsprop', 'sgd'], default='adam')
    parser.add_argument('--scheduler', type=eval, choices=[True, False], default=False)
    parser.add_argument('--num_iters', type=int, default=1000)
    parser.add_argument('--batch_size', help='Minibatch size', type=int, default=64)
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--annealing-iters', type=int, default=0)
    parser.add_argument('--save', help='directory to save results', type=str, default='figures/experiment1')
    parser.add_argument('--val-batchsize', help='minibatch size', type=int, default=200)
    parser.add_argument('--validation', type=eval, choices=[True, False], default=True)
    parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
    parser.add_argument('--update-freq', type=int, default=1)

    parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
    parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
    parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
    parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')

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
