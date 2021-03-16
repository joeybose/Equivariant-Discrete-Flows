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

def toy_flow(n_blocks, input_dim, hidden_dim, num_layers):
    base_dist = StandardNormal(shape=[input_dim])
    transforms = []
    n_blocks = 1 # Not Used
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim,
                                                              hidden_features=hidden_dim))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)
    return flow

def package_realnvp(n_blocks, input_dim, hidden_dim, num_layers):
    flow = realnvp.SimpleRealNVP(features=input_dim,
                                 hidden_features=hidden_dim,
                                 num_layers=num_layers,
                                 num_blocks_per_layer=n_blocks)
    return flow

# All code below this line is taken from
# https://github.com/kamenbliznashki/normalizing_flows/blob/master/maf.py

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
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
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
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden,
                 layer_type='Linear', radius=torch.Tensor([0])):
        super(RealNVP, self).__init__()
        mask = torch.arange(input_size).float() % 2
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.radius = radius
        self.layer_type = layer_type
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(n_blocks/2) + 1, 1)
        self.p_z = StandardNormal
        self.s, self.t = create_real_nvp_blocks(input_size, hidden_size,
                                                n_blocks, n_hidden, layer_type)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.mask = nn.Parameter(mask, requires_grad=False)

    def inverse(self, z, edge_index=None):
        log_det_J, x = z.new_zeros(z.shape[0]), z
        for i in range(0,self.n_blocks):
            x_ = x*self.mask[i]
            if self.layer_type != 'Linear':
                s = self.s[i](x_, edge_index)
                t = self.t[i](x_, edge_index)
            else:
                s = self.s[i](x_)
                t = self.t[i](x_)
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
            log_det_J += ((1-self.mask[i])*s).sum(dim=1)  # log det dx/du
        return x, log_det_J

    def forward(self, x, edge_index=None):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(0,self.n_blocks)):
            z_ = self.mask[i] * z
            if self.layer_type != 'Linear':
                s = self.s[i](z_, edge_index)
                t = self.t[i](z_, edge_index)
            else:
                s = self.s[i](z_)
                t = self.t[i](z_)
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= ((1-self.mask[i])*s).sum(dim=1)
        return z, log_det_J

    def log_prob(self, inputs, edge_index=None):
        z, logp = self.forward(inputs, edge_index)
        p_z = self.p_z([inputs.shape[-1]])
        return p_z.log_prob(z) + logp

    def sample(self, batchSize):
        # TODO: Update this method for edge_index
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x


