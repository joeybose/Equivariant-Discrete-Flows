import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions
from torch.nn.parameter import Parameter
import ipdb
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from flows.flow_helpers import *

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.flows import realnvp

import flows.layers.base as base_layers
import flows.layers as layers

ACTIVATION_FNS = {
    'relu': torch.nn.ReLU,
    'tanh': torch.nn.Tanh,
    'elu': torch.nn.ELU,
    'selu': torch.nn.SELU,
    'fullsort': base_layers.FullSort,
    'maxmin': base_layers.MaxMin,
    'swish': base_layers.Swish,
    'lcube': base_layers.LipschitzCube,
}

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def add_padding(args, x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


#Reference: https://github.com/ritheshkumar95/pytorch-normalizing-flows/blob/master/modules.py
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def toy_flow(args, n_blocks, input_dim, hidden_dim, num_layers):
    base_dist = StandardNormal(shape=[input_dim])
    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim,
                                                              hidden_features=hidden_dim))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    return flow

def package_realnvp(args, n_blocks, input_dim, hidden_dim, num_layers):
    flow = realnvp.SimpleRealNVP(features=input_dim,
                                 hidden_features=hidden_dim,
                                 num_layers=num_layers,
                                 num_blocks_per_layer=n_blocks)
    return flow


# All code below this line is taken from
# https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py
## Taken from: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb

class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        i = len(self)
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
            i -= 1
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        i = 0
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
            i += 1
        return u, sum_log_abs_det_jacobians

# --------------------
# Models
# --------------------

class MAFRealNVP(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden,
                 radius=torch.Tensor([0]), cond_label_size=None, batch_norm=False):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.p_z = StandardNormal
        self.radius = radius

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            # modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, inputs, y=None):
        u, sum_log_abs_det_jacobians = self.forward(inputs, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)

## Taken from: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class RealNVP(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden,
                 layer_type='Conv'):
        super(RealNVP, self).__init__()
        _, self.c, self.h, self.w = input_size[:]
        # mask_size = self.c * self.h * self.w
        # mask = torch.arange(mask_size).float() % 2
        self.n_blocks = int(n_blocks)
        self.n_hidden = n_hidden
        self.layer_type = layer_type
        checkerboard = [[((i % 2) + j) % 2 for j in range(self.w)] for i in range(self.h)]
        mask = torch.tensor(checkerboard).float()
        # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
        mask = mask.view(1, 1, self.h, self.w)
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i_mask = 1 - mask
        mask = torch.vstack([mask,i_mask]).repeat(int(self.n_blocks/2), 1, 1, 1)
        self.p_z = StandardNormal
        self.s, self.t = create_real_nvp_blocks(self.c, hidden_size,
                                                self.n_blocks, n_hidden, layer_type)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def inverse(self, z, logpz=None):
        z = z.view(-1, self.c, self.h, self.w)
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(0,self.n_blocks):
            x_ = x*self.mask[i]
            s = self.s[i](x_)
            t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += ((1-self.mask[i])*s).sum(dim=(1,2,3))  # log det dx/du
        return x.squeeze() if logpz is None else (z, -1*log_det_J.view(-1,1))

    def forward(self, x, inverse=False):
        if inverse:
            return self.inverse(x)

        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(0,self.n_blocks)):
            z_ = self.mask[i] * z
            s = self.s[i](z_)
            t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= ((1-self.mask[i])*s).sum(dim=(1,2,3))
        return z.squeeze(), log_det_J.view(-1, 1)

    def log_prob(self, inputs, beta=1.):
        z, delta_logp = self.forward(inputs)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
        logpx = logpz - beta * delta_logp
        return logpx, logpz, -1*delta_logp
        # p_z = self.p_z([inputs.shape[-1]])
        # return p_z.log_prob(z) + logp

    def compute_loss(self, args, inputs, beta=1.):
        bits_per_dim, logits_tensor = torch.zeros(1).to(inputs), torch.zeros(args.n_classes).to(inputs)
        logpz, delta_logp = torch.zeros(1).to(inputs), torch.zeros(1).to(inputs)

        if args.dataset == 'celeba_5bit':
            nvals = 32
        elif args.dataset == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256

        padded_inputs, logpu = add_padding(args, inputs, nvals)
        _, logpz, delta_logp = self.log_prob(padded_inputs, beta)

        # log p(x)
        logpx = logpz - beta * delta_logp - np.log(nvals) * (
            args.imagesize * args.imagesize * (args.im_dim + args.padding)
        ) - logpu
        bits_per_dim = -torch.mean(logpx) / (args.imagesize *
                                             args.imagesize * args.im_dim) / np.log(2)

        logpz = torch.mean(logpz).detach()
        delta_logp = torch.mean(-delta_logp).detach()
        return bits_per_dim, logits_tensor, logpz, delta_logp

    def sample(self, batchSize):
        # TODO: Update this method for edge_index
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x


