import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
import sklearn.datasets as datasets

import math
import os
import time
import argparse
import pprint
from functools import partial
import ipdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm
# from utils import utils
# from utils.utils import seed_everything, str2bool, visualize_transform
from utils.utils import *
from data import create_dataset
from data.toy_data import sample_2d_data
from flows import create_flow
from e2cnn import gspaces
from e2cnn import nn as enn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test(args, epoch, flow, whole_data, test_data_iter, prior):
    flow.update_lipschitz(200)
    holdout_nll = 0
    with torch.no_grad():
        flow.eval()
        for it, batch_idxs in enumerate(test_data_iter):
            if it > 100:
                break
            x = torch.Tensor(whole_data[batch_idxs]).cuda()
            test_loss, test_logpz, test_delta_logp = flow.log_prob(inputs=x, prior=prior)
            # z, delta_logp = flow(x, inverse=True)
            # nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()
            print("\r{}".format(it), test_loss.item(), end="")
            holdout_nll += test_loss.item()
        holdout_nll = holdout_nll / (it + 1)

        print(
            '[TEST] Iter {:04d} | Test NLL {:.6f} '.format(
                epoch, holdout_nll
            )
        )
        flow.train()

def train_flow(args, flow, optimizer):
    time_meter = RunningAverageMeter(0.93)
    loss_meter = RunningAverageMeter(0.93)
    logpz_meter = RunningAverageMeter(0.93)
    delta_logp_meter = RunningAverageMeter(0.93)
    end = time.time()
    flow.train()
    data_smaller, whole_data, train_data_iter, test_data_iter, prior = create_dataset(args, args.dataset)
    # if args.use_whole_data:
        # data = torch.tensor(whole_data, dtype=torch.float32, device=args.dev)
    # else:
        # data = torch.tensor(data_smaller, dtype=torch.float32, device=args.dev)
    for i in range(args.num_iters):
        for it, idx in enumerate(train_data_iter):
            data = torch.tensor(whole_data[idx], dtype=torch.float32,
                                device=args.dev)
            optimizer.zero_grad()
            beta = min(1, itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
            loss, logpz, delta_logp = flow.log_prob(inputs=data, prior=prior)
            try:
                if len(logpz) > 0:
                    logpz = torch.mean(logpz)
                    delta_logp = torch.mean(delta_logp)
            except:
                pass
            # loss = -flow.log_prob(inputs=data).mean()
            loss_meter.update(loss.item())
            logpz_meter.update(logpz.item())
            delta_logp_meter.update(delta_logp.item())

            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(flow.parameters(), 1.)
            optimizer.step()
            if 'resflow' in args.model_type:
                flow.beta = beta
                flow.update_lipschitz(args.n_lipschitz_iters)
                if args.learn_p and itr > args.annealing_iters: flow.compute_p_grads()

            time_meter.update(time.time() - end)
            print(
                'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f})'
                ' | Logp(z) {:.6f}({:.6f}) | DeltaLogp {:.6f}({:.6f})'.format(
                    i+1, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, logpz_meter.val, logpz_meter.avg,
                    delta_logp_meter.val, delta_logp_meter.avg
                )
            )

            if (i + 1) % args.test_interval == 0 or i == args.num_iters:
                test(args, i, flow, whole_data, test_data_iter, prior)
                # ipdb.set_trace()

            if (i + 1) % args.log_interval == 0 and args.plot:
                print("Log Likelihood at %d is %f" %(i+1, loss))
                flow.update_lipschitz(200)
                with torch.no_grad():
                    flow.eval()
                    p_samples = sample_2d_data(args.dataset, 400)
                    # sample_fn, density_fn = flow.flow_model.inverse, flow.flow_model.forward
                    sample_fn, density_fn = None, flow.flow_model.forward

                    plt.figure(figsize=(9, 3))
                    visualize_transform(
                        p_samples, torch.randn, flow.standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
                        samples=True, npts=100, device=args.dev
                    )
                    plt.savefig('figures/E_figures/{}_{}.png'.format(args.plot_name, str(i+1)))
                    plt.close()
                    flow.train()

            end = time.time()


def main(args):
    # args.input_size = (args.batch_size, 1, 1, args.input_dim)
    args.input_size = (args.batch_size, args.nc, 1, args.input_dim)
    flow = create_flow(args, args.model_type)
    print("Number of trainable parameters: {}".format(count_parameters(flow)))
    # optimizer = optim.Adam(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.AdamW(flow.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
    train_flow(args, flow, optimizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # boiler plate inits
    parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
    parser.add_argument('--cuda', type=int, help='Which GPU to run on.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--num_iters', type=int, default=500)
    parser.add_argument('--plot_name', type=str, default='test')
    # target density
    parser.add_argument('--model_type', type=str, default="Toy", help='Which Flow to use.')
    parser.add_argument('--dataset', type=str, default=None, help='Which potential function to approximate.')
    # parser.add_argument('--nsamples', type=int, default=500, help='Number of Samples to Use')
    # model parameters
    parser.add_argument('--input_size', type=int, default=2, help='Dimension of the data.')
    parser.add_argument('--input_dim', type=int, default=2, help='Dimension of the data.')
    parser.add_argument('--nc', type=int, default=1, help='Num channels.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Dimensions of hidden layers.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--n_blocks', type=str, default='1')
    # i-Resnet params
    parser.add_argument('--coeff', type=float, default=0.9)
    parser.add_argument('--vnorms', type=str, default='222222')
    parser.add_argument('--n-lipschitz-iters', type=int, default=100)
    parser.add_argument('--atol', type=float, default=None)
    parser.add_argument('--rtol', type=float, default=None)
    parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)
    parser.add_argument('--mixed', type=eval, choices=[True, False], default=True)
    parser.add_argument('--mean-free-prior', type=eval, choices=[True, False], default=False)
    parser.add_argument('--beta', type=float, default=1.0)

    parser.add_argument('--dims', type=str, default='128-128-128-128')
    parser.add_argument('--act', type=str, default='swish')
    parser.add_argument('--group', type=str, default='fliprot4', help='The choice of group representation for Equivariance')
    parser.add_argument('--out-fiber', type=str, default='regular')
    parser.add_argument('--field-type', type=int, default=0, help='Only For Continuous groups. Picks the frequency.')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--realnvp-padding', type=int, default=1)
    parser.add_argument('--brute-force', type=eval, choices=[True, False], default=False)
    parser.add_argument('--actnorm', type=eval, choices=[True, False], default=False)
    parser.add_argument('--batchnorm', type=eval, choices=[True, False], default=False)
    parser.add_argument('--exact-trace', type=eval, choices=[True, False], default=False)
    parser.add_argument('--n-power-series', type=int, default=None)
    parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')
    # training parameters
    parser.add_argument('--use_whole_data', type=eval, choices=[True, False], default=False)
    parser.add_argument('--log_interval', type=int, default=10, help='How often to save model and samples.')
    parser.add_argument('--test_interval', type=int, default=500, help='How often to save model and samples.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--annealing-iters', type=int, default=0)

    args = parser.parse_args()

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name = project_name(args.dataset)
    main(args)
