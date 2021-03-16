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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import utils
from utils.utils import seed_everything, str2bool
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from data import create_dataset


def train_flow(args, flow, optim, data=None):
    num_iter = 5000
    for i in range(num_iter):
        if data is None:
            data, y = datasets.make_moons(128, noise=.1)
            data = torch.tensor(data, dtype=torch.float32).to(args.dev)
        optim.zero_grad()
        loss = -flow.log_prob(inputs=data).mean()
        loss.backward()
        optim.step()
        if (i + 1) % args.log_interval == 0 and args.plot:
            print("Log Likelihood at %d is %f" %(i+1, loss))
            xline = torch.linspace(-3, 3, steps=100)
            yline = torch.linspace(-3, 3, steps=100)
            xgrid, ygrid = torch.meshgrid(xline, yline)
            xyinput = torch.cat([xgrid.reshape(-1, 1), ygrid.reshape(-1, 1)],
                                dim=1).to(args.dev)

            with torch.no_grad():
                zgrid = flow.log_prob(xyinput).exp().reshape(100, 100)

            plt.contourf(xgrid.numpy(), ygrid.numpy(), zgrid.cpu().numpy())
            plt.title('iteration {}'.format(i + 1))
            plt.show()
            plt.savefig('figures/{}_{}.png'.format(args.plot_name, str(i+1)))


def main(args):
    # Define an invertible transformation.
    base_dist = StandardNormal(shape=[2])
    data = None
    transforms = []
    for _ in range(args.num_layers):
        transforms.append(ReversePermutation(features=2))
        transforms.append(MaskedAffineAutoregressiveTransform(features=2,
                                                              hidden_features=args.hidden_dim))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist).to(args.dev)
    optimizer = optim.Adam(flow.parameters())
    if args.dataset is not None:
        data = create_dataset(args, args.dataset).to(args.dev)
    train_flow(args, flow, optimizer, data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # boiler plate inits
    parser.add_argument('--plot', action='store_true', help='Plot a flow and target density.')
    parser.add_argument('--cuda', type=int, help='Which GPU to run on.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--plot_name', type=str, default='test')
    # target density
    parser.add_argument('--dataset', type=str, default=None, help='Which potential function to approximate.')
    parser.add_argument('--nsamples', type=int, default=500, help='Number of Samples to Use')
    # model parameters
    parser.add_argument('--data_dim', type=int, default=2, help='Dimension of the data.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Dimensions of hidden layers.')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of hidden layers.')
    # training parameters
    parser.add_argument('--log_interval', type=int, default=500, help='How often to save model and samples.')

    args = parser.parse_args()

    ''' Fix Random Seed '''
    seed_everything(args.seed)
    # Check if settings file

    args.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_name = utils.project_name(args.dataset)


    main(args)
