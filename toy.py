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
from utils import utils
from utils.utils import seed_everything, str2bool, visualize_transform
from data import create_dataset
from data.toy_data import sample_2d_data
from flows import create_flow
from e2cnn import gspaces
from e2cnn import nn as enn

def train_flow(args, flow, optim):
    # if args.dataset is None:
        # data, y = datasets.make_moons(128, noise=.1)
        # data = torch.tensor(data, dtype=torch.float32).to(args.dev)
    # else:
        # data = create_dataset(args, args.dataset).to(args.dev)
    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    logpz_meter = utils.RunningAverageMeter(0.93)
    delta_logp_meter = utils.RunningAverageMeter(0.93)
    end = time.time()
    flow.train()
    for i in range(args.num_iters):
        if args.dataset is None:
            data, y = datasets.make_moons(128, noise=.1)
            data = torch.tensor(data, dtype=torch.float32).to(args.dev)
        else:
            data = create_dataset(args, args.dataset)
            data = torch.from_numpy(data).type(torch.float32).to(args.dev)
        optim.zero_grad()
        beta = min(1, itr / args.annealing_iters) if args.annealing_iters > 0 else 1.
        loss, logpz, delta_logp = flow.log_prob(inputs=data)
        # loss = -flow.log_prob(inputs=data).mean()
        loss_meter.update(loss.item())
        logpz_meter.update(logpz.item())
        delta_logp_meter.update(delta_logp.item())

        loss.backward()
        optim.step()

        if args.model_type == 'toy_resflow':
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


        if (i + 1) % args.log_interval == 0 and args.plot:
            print("Log Likelihood at %d is %f" %(i+1, loss))
            with torch.no_grad():
                flow.eval()
                p_samples = sample_2d_data(args.dataset, 20000)
                sample_fn, density_fn = flow.flow_model.inverse, flow.flow_model.forward

                plt.figure(figsize=(9, 3))
                visualize_transform(
                    p_samples, torch.randn, flow.standard_normal_logprob, transform=sample_fn, inverse_transform=density_fn,
                    samples=True, npts=400, device=args.dev
                )
                plt.savefig('figures/{}_{}.png'.format(args.plot_name, str(i+1)))
                plt.close()
                flow.train()

        end = time.time()


def main(args):
    flow = create_flow(args, args.model_type)
    optimizer = optim.Adam(flow.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    parser.add_argument('--nsamples', type=int, default=500, help='Number of Samples to Use')
    # model parameters
    parser.add_argument('--input_dim', type=int, default=2, help='Dimension of the data.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Dimensions of hidden layers.')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of hidden layers.')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks.')
    # i-Resnet params
    parser.add_argument('--coeff', type=float, default=0.9)
    parser.add_argument('--vnorms', type=str, default='222222')
    parser.add_argument('--n-lipschitz-iters', type=int, default=5)
    parser.add_argument('--atol', type=float, default=None)
    parser.add_argument('--rtol', type=float, default=None)
    parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)
    parser.add_argument('--mixed', type=eval, choices=[True, False], default=True)
    parser.add_argument('--beta', type=float, default=1.0)

    parser.add_argument('--dims', type=str, default='128-128-128-128')
    parser.add_argument('--act', type=str, default='swish')
    parser.add_argument('--brute-force', type=eval, choices=[True, False], default=False)
    parser.add_argument('--actnorm', type=eval, choices=[True, False], default=False)
    parser.add_argument('--batchnorm', type=eval, choices=[True, False], default=False)
    parser.add_argument('--exact-trace', type=eval, choices=[True, False], default=False)
    parser.add_argument('--n-power-series', type=int, default=None)
    parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='geometric')
    # training parameters
    parser.add_argument('--log_interval', type=int, default=10, help='How often to save model and samples.')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--test_batch_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--annealing-iters', type=int, default=0)

    args = parser.parse_args()

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name = utils.project_name(args.dataset)


    main(args)
