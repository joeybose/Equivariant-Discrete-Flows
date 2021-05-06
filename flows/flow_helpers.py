import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
import torch.nn.init as init

# from utils.utils import MultiInputSequential
from utils import utils
import math
import os
import math
import argparse
import pprint
import numpy as np
import copy
import flows.layers.base as base_layers
import flows.layers as layers
from flows.invariant_maps import InvariantCNNBlock, InvariantCNNBlock2, InvariantCNNBlock3
import ipdb
from e2cnn import gspaces
from e2cnn import nn as enn
# --------------------
# Model layers and helpers
# --------------------

# FIBERS = {
    # "trivial": utils.trivial_fiber,
    # "quotient": utils.quotient_fiber,
    # "regular": utils.regular_fiber,
    # "irrep": utils.irrep_fiber,
    # "mixed1": utils.mixed1_fiber,
    # "mixed2": utils.mixed2_fiber,
# }

kwargs_layer = {'Linear': nn.Linear, 'Conv': nn.Conv2d}

class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        multi_inp = False
        if len(input) > 1:
            multi_inp = True
            _, other_inp = input[0], input[1]

        for module in self._modules.values():
            if multi_inp:
                if hasattr(module, 'weight') or hasattr(module, 'weights'):
                    input = [module(*input)]
                else:
                    input = [module(input[0]), other_inp]
            else:
                input = [module(*input)]
        return input[0]

def create_real_nvp_blocks(input_size, hidden_size, n_blocks, n_hidden,
                          layer_type='Linear'):
    nets, nett = [], []
    # Build the Flow Block by Block
    if layer_type == 'Conv':
        kernel_size = 3
        stride = 1
        padding = 1

    for i in range(n_blocks):
        block_nets = [kwargs_layer[layer_type](input_size, hidden_size,
                                               kernel_size, stride, padding)]
        block_nett = [kwargs_layer[layer_type](input_size, hidden_size,
                                               kernel_size, stride, padding)]
        for _ in range(n_hidden):
            block_nets += [nn.Tanh(), kwargs_layer[layer_type](hidden_size,
                                                               hidden_size,
                                                               kernel_size,
                                                               stride,
                                                               padding)]
            block_nett += [nn.Tanh(), kwargs_layer[layer_type](hidden_size,
                                                               hidden_size,
                                                               kernel_size,
                                                               stride,
                                                               padding)]
        block_nets += [nn.Tanh(), kwargs_layer[layer_type](hidden_size,
                                                           input_size,
                                                           kernel_size,
                                                           stride,
                                                           padding)]
        block_nett += [nn.Tanh(), kwargs_layer[layer_type](hidden_size,
                                                           input_size,
                                                           kernel_size,
                                                           stride,
                                                           padding)]
        nets +=[MultiInputSequential(*block_nets)]
        nett +=[MultiInputSequential(*block_nett)]

    s = nets = MultiInputSequential(*nets)
    t = nett = MultiInputSequential(*nett)
    return s,t


def create_invariant_real_nvp_blocks(input_size, in_type, field_type,
                                       out_fiber, activation_fn, hidden_size,
                                       n_blocks, n_hidden, group_action_type,
                                       kernel_size=3, padding=1, only_t=False):
    nets = []

    # we store the input type for wrapping the images into a geometric tensor during the forward pass
    input_type = in_type
    _, c, h , w = input_size
    out_type = enn.FieldType(group_action_type, c*[group_action_type.trivial_repr])
    inter_block_out_type = FIBERS[out_fiber](group_action_type, hidden_size,
                                      field_type, fixparams=True)
    for i in range(n_blocks):
        # s_block = InvariantCNNBlock(input_size, in_type, field_type,
                                      # out_fiber, activation_fn, hidden_size,
                                      # group_action_type)
        s_block = InvariantCNNBlock3(input_size, in_type, field_type,
                                      out_fiber, activation_fn, hidden_size,
                                      group_action_type)
        nets +=[s_block]

    s = nets = MultiInputSequential(*nets)
    return s

def create_equivariant_real_nvp_blocks(input_size, in_type, field_type,
                                       out_fiber, activation_fn, hidden_size,
                                       n_blocks, n_hidden, group_action_type,
                                       kernel_size=3, padding=1, only_t=False):
    nets, nett = [], []

    # we store the input type for wrapping the images into a geometric tensor during the forward pass
    input_type = in_type
    _, c, h , w = input_size
    out_type = enn.FieldType(group_action_type, c*[group_action_type.trivial_repr])
    inter_block_out_type = FIBERS[out_fiber](group_action_type, hidden_size,
                                      field_type, fixparams=True)
    for i in range(n_blocks):
        if not only_t:
            s_block = [enn.SequentialModule(
                enn.R2Conv(in_type, inter_block_out_type, kernel_size=kernel_size,
                           padding=padding, bias=True),
                enn.InnerBatchNorm(inter_block_out_type),
                activation_fn(inter_block_out_type, inplace=True)
            )]
        t_block = [enn.SequentialModule(
            enn.R2Conv(in_type, inter_block_out_type, kernel_size=kernel_size,
                       padding=padding, bias=True),
            enn.InnerBatchNorm(inter_block_out_type),
            activation_fn(inter_block_out_type, inplace=True)
        )]
        for _ in range(n_hidden):
            if not only_t:
                s_block += [enn.SequentialModule(
                    enn.R2Conv(s_block[-1].out_type, inter_block_out_type,
                               kernel_size=kernel_size, padding=padding, bias=True),
                    enn.InnerBatchNorm(inter_block_out_type),
                    activation_fn(inter_block_out_type, inplace=True)
                )]
            t_block += [enn.SequentialModule(
                enn.R2Conv(t_block[-1].out_type, inter_block_out_type,
                           kernel_size=kernel_size, padding=padding, bias=True),
                enn.InnerBatchNorm(inter_block_out_type),
                activation_fn(inter_block_out_type, inplace=True)
            )]

        if not only_t:
            s_block += [enn.SequentialModule(
                enn.R2Conv(s_block[-1].out_type, in_type, kernel_size=kernel_size,
                           padding=padding, bias=True),
                enn.InnerBatchNorm(out_type),
                activation_fn(out_type, inplace=True)
            )]
            nets +=[MultiInputSequential(*s_block)]

        t_block += [enn.SequentialModule(
            enn.R2Conv(t_block[-1].out_type, in_type, kernel_size=kernel_size,
                       padding=padding, bias=True),
            enn.InnerBatchNorm(out_type),
            activation_fn(out_type, inplace=True)
        )]
        nett +=[MultiInputSequential(*t_block)]

    t = nett = MultiInputSequential(*nett)
    if not only_t:
        s = nets = MultiInputSequential(*nets)
        return s,t
    else:
        return t


def create_equivariant_convexp_blocks(input_size, hidden_size, n_blocks,
                                       n_hidden, group_action_type,
                                       kernel_size=3, padding=1):
    nets = []
    # the model is equivariant under rotations by 45 degrees, modelled by C8

    # the input image is a scalar field, corresponding to the trivial representation
    in_type = enn.FieldType(group_action_type, [group_action_type.trivial_repr])

    # we store the input type for wrapping the images into a geometric tensor during the forward pass
    input_type = in_type

    out_type = enn.FieldType(group_action_type, [group_action_type.trivial_repr])
    # out_type = enn.FieldType(group_action_type, hidden_size*[group_action_type.regular_repr])
    for i in range(n_blocks):
        # s_block = [enn.SequentialModule(
            # layers.R2EquivariantConv(in_type=in_type, out_type=out_type, kernel_size=kernel_size,
                       # padding=padding, bias=True),
            # # enn.InnerBatchNorm(out_type),
            # # enn.ELU(out_type, inplace=True)
        # )]
        s_block = [
            layers.R2EquivariantConv(in_type=in_type, out_type=out_type, kernel_size=kernel_size,
                       padding=padding, bias=True),
            # enn.InnerBatchNorm(out_type),
            enn.ELU(out_type, inplace=True)
        ]
        # inter_block_out_type = enn.FieldType(group_action_type, hidden_size*[group_action_type.regular_repr])
        # inter_block_out_type = enn.FieldType(group_action_type, [group_action_type.trivial_repr])
        # for _ in range(n_hidden):
            # s_block += [enn.SequentialModule(
                # layers.R2EquivariantConv(s_block[-1].out_type, out_type=inter_block_out_type,
                           # kernel_size=kernel_size, padding=padding, bias=True),
                # # enn.InnerBatchNorm(inter_block_out_type),
                # enn.ELU(inter_block_out_type, inplace=True)
            # )]

        # s_block += [enn.SequentialModule(
            # layers.R2EquivariantConv(s_block[-1].out_type, out_type=in_type, kernel_size=kernel_size,
                       # padding=padding, bias=True),
            # # enn.InnerBatchNorm(out_type),
            # enn.ELU(in_type, inplace=True)
        # )]
        nets += [MultiInputSequential(*s_block)]

    s = nets = MultiInputSequential(*nets)
    return s


def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """ MADE building block layer """
    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """
    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(-s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian

class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0) # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
#        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
#            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = y.mean(0)
            var = y.var(0)
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)

# Taken from https://github.com/ehoogeboom/convolution_exponential_and_sylvester/blob/4103cdf8a8c080c4f68fe8acde6c702bd8dfe192/models/transformations/convexp/functional.py
def matrix_log(B, terms=10):
    assert B.size(0) == B.size(1)
    I = torch.eye(B.size(0))

    B_min_I = B - I

    # for k = 1.
    product = B_min_I
    result = B_min_I

    is_minus = -1
    for k in range(2, terms):
        # Reweighing with k term.
        product = torch.matmul(product, B_min_I) * (k - 1) / k
        result = result + is_minus * product

        is_minus *= -1

    return result


def matrix_exp(M, terms=10):
    assert M.size(0) == M.size(1)
    I = torch.eye(M.size(0))

    # for i = 0.
    result = I
    product = I

    for i in range(1, terms + 1):
        product = torch.matmul(product, M) / i
        result = result + product

    return result


def conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    B, C, H, W = input.size()

    assert kernel.size(0) == kernel.size(1)
    assert kernel.size(0) == C, '{} != {}'.format(kernel.size(0), C)

    padding = (kernel.size(2) - 1) // 2, (kernel.size(3) - 1) // 2

    result = input
    product = input

    for i in range(1, terms + 1):
        product = F.conv2d(product, kernel, padding=padding, stride=(1, 1)) / i
        result = result + product

        if dynamic_truncation != 0 and i > 5:
            if product.abs().max().item() < dynamic_truncation:
                break

    if verbose:
        print(
            'Maximum element size in term: {}'.format(
                torch.max(torch.abs(product))))

    return result


def inv_conv_exp(input, kernel, terms=10, dynamic_truncation=0, verbose=False):
    return conv_exp(input, -kernel, terms, dynamic_truncation, verbose)


def log_det(kernel):
    Cout, C, K1, K2 = kernel.size()
    assert Cout == C

    M1 = (K1 - 1) // 2
    M2 = (K2 - 1) // 2

    diagonal = kernel[torch.arange(C), torch.arange(C), M1, M2]

    trace = torch.sum(diagonal)

    return trace


def convergence_scale(c, kernel_size):
    C_out, C_in, K1, K2 = kernel_size

    d = C_in * K1 * K2

    return c / d
