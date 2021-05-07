import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
import ipdb
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler
from nflows.distributions.normal import StandardNormal
from nflows.distributions.uniform import BoxUniform
from utils import utils
from e2cnn import gspaces
from e2cnn import nn as enn
from flows.flow_helpers import *
import flows.layers.base as base_layers
import flows.layers as layers

ACT_FNS = {
    'relu': enn.ReLU,
    'elu': enn.ELU,
    'gated': enn.GatedNonLinearity1,
    'swish': base_layers.GeomSwish,
}

GROUPS = {
    'fliprot16': gspaces.FlipRot2dOnR2(N=16),
    'fliprot12': gspaces.FlipRot2dOnR2(N=12),
    'fliprot8': gspaces.FlipRot2dOnR2(N=8),
    'fliprot4': gspaces.FlipRot2dOnR2(N=4),
    'fliprot2': gspaces.FlipRot2dOnR2(N=2),
    'flip': gspaces.Flip2dOnR2(),
    'rot16': gspaces.Rot2dOnR2(N=16),
    'rot12': gspaces.Rot2dOnR2(N=12),
    'rot8': gspaces.Rot2dOnR2(N=8),
    'rot4': gspaces.Rot2dOnR2(N=4),
    'rot2': gspaces.Rot2dOnR2(N=2),
    'so2': gspaces.Rot2dOnR2(N=-1, maximum_frequency=10),
    'o2': gspaces.FlipRot2dOnR2(N=-1, maximum_frequency=10),
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
    elif args.double_padding:
        x = x.repeat(1, 2, 1, 1)
        return x, torch.zeros(x.shape[0], 1).to(x)
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)


class EquivariantRealNVP(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden,
                 group_action_type=None):
        super(EquivariantRealNVP, self).__init__()
        _, self.c, self.h, self.w = input_size[:]
        assert self.c > 1
        mask = torch.arange(self.c).float() % 2
        self.n_blocks = int(n_blocks)
        self.n_hidden = n_hidden
        self.group_action_type = GROUPS[args.group]
        self.out_fiber = args.out_fiber
        self.field_type = args.field_type
        self.group_card = len(list(self.group_action_type.testing_elements))
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(self.n_blocks/2) + 1, 1)
        self.p_z = StandardNormal
        self.input_type = enn.FieldType(self.group_action_type, self.c*[self.group_action_type.trivial_repr])
        self.activation_fn = ACT_FNS[args.act]
        self.t = create_equivariant_real_nvp_blocks(input_size,
                                                    self.input_type,
                                                    self.field_type,
                                                    self.out_fiber,
                                                    self.activation_fn,
                                                    hidden_size, self.n_blocks,
                                                    n_hidden,
                                                    self.group_action_type,
                                                    args.kernel_size,
                                                    args.realnvp_padding,
                                                    only_t=True)

        self.s = create_invariant_real_nvp_blocks(input_size,
                                                    self.input_type,
                                                    self.field_type,
                                                    self.out_fiber,
                                                    self.activation_fn,
                                                    hidden_size, self.n_blocks,
                                                    n_hidden,
                                                    self.group_action_type,
                                                    args.kernel_size,
                                                    args.realnvp_padding)

        self.mask = nn.Parameter(mask, requires_grad=False)

    def inverse(self, z, logpz=None):
        log_det_J, x = z.new_zeros(z.shape[0]), z.view(-1, self.c, self.h, self.w)
        for i in range(0,self.n_blocks):
            fiber_batch_mask = self.mask[i].view(self.c, 1, 1).repeat(x.shape[0], 1 , 1, 1)
            x_ = fiber_batch_mask*x
            x_ = enn.GeometricTensor(x_, self.input_type)
            s = self.s[i](x_).tensor
            t = self.t[i](x_).tensor
            inverse_fiber_batch_mask = (1- self.mask[i]).view(self.c, 1, 1).repeat(x.shape[0], 1 , 1, 1)
            x = x_.tensor + inverse_fiber_batch_mask * (x * torch.exp(s) + t)
            log_det_J += (inverse_fiber_batch_mask*s).sum(dim=(1,2,3))  # log det dx/du
        return x.squeeze() if logpz is None else (z, -1*log_det_J.view(-1,1))

    def dummy_func(self, z):
        for i in reversed(range(0,self.n_blocks)):
            z = z.tensor
            fiber_batch_mask = self.mask[i].view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z_ = fiber_batch_mask * z
            z_ = enn.GeometricTensor(z_, self.input_type)
            s = self.s[i](z_).tensor
            t = self.t[i](z_).tensor
            inverse_fiber_batch_mask = (1- self.mask[i]).view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z = inverse_fiber_batch_mask * (z - t) * torch.exp((-1.* s)) + z_.tensor
            z = enn.GeometricTensor(z, self.input_type)
        return z

    def forward(self, x, inverse=False):
        if inverse:
            return self.inverse(x)
        log_det_J, z = x.new_zeros(x.shape[0]), x.view(-1, self.c, self.h, self.w)
        # my_x = enn.GeometricTensor(x, self.input_type)
        # self.check_equivariance(self.group_action_type, self.input_type, my_x,
                                # self.dummy_func)
        for i in reversed(range(0,self.n_blocks)):
            fiber_batch_mask = self.mask[i].view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z_ = fiber_batch_mask * z
            z_ = enn.GeometricTensor(z_, self.input_type)
            # ipdb.set_trace()
            # self.check_invariance(self.group_action_type, self.input_type,
                                  # z_, self.s[i])
            # self.check_equivariance(self.group_action_type, self.input_type,
                                  # z_, self.s[i])
            s = self.s[i](z_).tensor
            t = self.t[i](z_).tensor
            inverse_fiber_batch_mask = (1- self.mask[i]).view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z = inverse_fiber_batch_mask * (z - t) * torch.exp((-1.* s)) + z_.tensor
            log_det_J -= (inverse_fiber_batch_mask*s).sum(dim=(1,2,3))
        return z.squeeze(), log_det_J.view(-1, 1)

    def log_prob(self, inputs, beta=1.):
        z, delta_logp = self.forward(inputs)
        # compute log p(z)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
        logpx = logpz - beta * delta_logp
        # loss = -torch.mean(logpx)
        # return loss, torch.mean(logpz), torch.mean(-delta_logp), None
        # print("NLL %f " % (-loss.item()))
        return logpx, logpz, -1*delta_logp

    def compute_loss(self, args, inputs, beta=1.):
        bits_per_dim, logits_tensor = torch.zeros(1).to(inputs), torch.zeros(args.n_classes).to(inputs)
        logpz, delta_logp = torch.zeros(1).to(inputs), torch.zeros(1).to(inputs)
        # z, _ = self.forward(inputs)
        # x = self.inverse(z)
        if args.dataset == 'celeba_5bit':
            nvals = 32
        elif args.dataset == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256

        # z, _ = self.forward(inputs)
        # x, _ = self.inverse(z)
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
        return bits_per_dim, logits_tensor, logpz, delta_logp, _

    def sample(self, batchSize):
        # TODO: Update this method for edge_index
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x

    def check_equivariance(self, r2_act, out_type, data, func, data_type=None):
        _, c, h, w = data.shape
        input_type = enn.FieldType(r2_act, self.c*[r2_act.trivial_repr])
        for g in r2_act.testing_elements:
            output = func(data)
            rg_output = enn.GeometricTensor(output.tensor.view(-1, c, h, w).cpu(),
                                         out_type).transform(g)
            data = enn.GeometricTensor(data.tensor.view(-1, c, h, w).cpu(), input_type)
            x_transformed = enn.GeometricTensor(data.transform(g).tensor.view(-1, c, h, w).cuda(), input_type)

            output_rg = func(x_transformed)
            # Equivariance Condition
            output_rg = enn.GeometricTensor(output_rg.tensor.cpu(), out_type)
            data = enn.GeometricTensor(data.tensor.squeeze().view(-1, c, h , w).cuda(), input_type)
            assert torch.allclose(rg_output.tensor.cpu().squeeze(), output_rg.tensor.squeeze(), atol=1e-5), g
        print("Passed Equivariance Test")

    def check_invariance(self, r2_act, out_type, data, func, data_type=None):
        _, c, h, w = data.shape
        input_type = enn.FieldType(r2_act, self.c*[r2_act.trivial_repr])
        y = func(data)
        for g in r2_act.testing_elements:
            data = enn.GeometricTensor(data.tensor.view(-1, c, h, w).cpu(), input_type)
            x_transformed = enn.GeometricTensor(data.transform(g).tensor.view(-1, c, h, w).cuda(), input_type)

            y_from_x_transformed = func(x_transformed)

            y_transformed_from_x = y
            # Invariance Condition
            data = enn.GeometricTensor(data.tensor.squeeze().view(-1, c, h , w).cuda(), input_type)
            # assert torch.allclose(output_conv.squeeze(), output_rg_conv.squeeze(), atol=1e-5), g
            print_y = y_from_x_transformed.tensor.detach().to('cpu').numpy().squeeze()

            print("{:4d} : {}".format(g, print_y))

            assert torch.allclose(y_from_x_transformed.tensor.squeeze(), y_transformed_from_x.tensor.squeeze(), atol=1e-5), g
        print("Passed Invariance Test")

## Inspired from: https://github.com/senya-ashukha/real-nvp-pytorch/blob/master/real-nvp-pytorch.ipynb
class FiberRealNVP(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden,
                 group_action_type=None):
        super(FiberRealNVP, self).__init__()
        _, self.c, self.h, self.w = input_size[:]
        assert self.c > 1
        mask = torch.arange(self.c).float() % 2
        self.n_blocks = int(n_blocks)
        self.n_hidden = n_hidden
        self.group_action_type = GROUPS[args.group]
        self.out_fiber = args.out_fiber
        self.field_type = args.field_type
        self.group_card = len(list(self.group_action_type.testing_elements))
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        i_mask = 1 - mask
        mask = torch.stack([mask,i_mask]).repeat(int(self.n_blocks/2) + 1, 1)
        self.p_z = StandardNormal
        self.input_type = enn.FieldType(self.group_action_type, self.c*[self.group_action_type.trivial_repr])
        self.activation_fn = ACT_FNS[args.act]
        self.s, self.t = create_equivariant_real_nvp_blocks(input_size,
                                                            self.input_type,
                                                            self.field_type,
                                                            self.out_fiber,
                                                            self.activation_fn,
                                                            hidden_size,
                                                            self.n_blocks, n_hidden,
                                                            self.group_action_type,
                                                            args.kernel_size,
                                                            args.realnvp_padding)
        self.mask = nn.Parameter(mask, requires_grad=False)

    def inverse(self, z, logpz=None):
        log_det_J, x = z.new_zeros(z.shape[0]), z.view(-1, self.c, self.h, self.w)
        for i in range(0,self.n_blocks):
            fiber_batch_mask = self.mask[i].view(self.c, 1, 1).repeat(x.shape[0], 1 , 1, 1)
            x_ = fiber_batch_mask*x
            x_ = enn.GeometricTensor(x_, self.input_type)
            s = self.s[i](x_).tensor
            t = self.t[i](x_).tensor
            inverse_fiber_batch_mask = (1- self.mask[i]).view(self.c, 1, 1).repeat(x.shape[0], 1 , 1, 1)
            x = x_.tensor + inverse_fiber_batch_mask * (x * torch.exp(s) + t)
            log_det_J += (inverse_fiber_batch_mask*s).sum(dim=(1,2,3))  # log det dx/du
        return x.squeeze() if logpz is None else (z, -1*log_det_J.view(-1,1))

    def dummy_func(self, z):
        for i in reversed(range(0,self.n_blocks)):
            z = z.tensor
            fiber_batch_mask = self.mask[i].view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z_ = fiber_batch_mask * z
            z_ = enn.GeometricTensor(z_, self.input_type)
            s = self.s[i](z_).tensor
            t = self.t[i](z_).tensor
            inverse_fiber_batch_mask = (1- self.mask[i]).view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z = inverse_fiber_batch_mask * (z - t) * torch.exp((-1.* s)) + z_.tensor
            z = enn.GeometricTensor(z, self.input_type)
        return z

    def forward(self, x, inverse=False):
        if inverse:
            return self.inverse(x)
        log_det_J, z = x.new_zeros(x.shape[0]), x.view(-1, self.c, self.h, self.w)
        # my_x = enn.GeometricTensor(x, self.input_type)
        # self.check_equivariance(self.group_action_type, self.input_type, my_x,
                                # self.dummy_func)
        for i in reversed(range(0,self.n_blocks)):
            fiber_batch_mask = self.mask[i].view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z_ = fiber_batch_mask * z
            z_ = enn.GeometricTensor(z_, self.input_type)
            s = self.s[i](z_).tensor
            t = self.t[i](z_).tensor
            inverse_fiber_batch_mask = (1- self.mask[i]).view(self.c, 1, 1).repeat(z.shape[0], 1 , 1, 1)
            z = inverse_fiber_batch_mask * (z - t) * torch.exp((-1.* s)) + z_.tensor
            log_det_J -= (inverse_fiber_batch_mask*s).sum(dim=(1,2,3))
        return z.squeeze(), log_det_J.view(-1, 1)

    def log_prob(self, inputs, beta=1.):
        z, delta_logp = self.forward(inputs)
        # compute log p(z)
        logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
        logpx = logpz - beta * delta_logp
        # loss = -torch.mean(logpx)
        # return loss, torch.mean(logpz), torch.mean(-delta_logp), None
        # print("NLL %f " % (-loss.item()))
        return logpx, logpz, -1*delta_logp

    def compute_loss(self, args, inputs, beta=1.):
        bits_per_dim, logits_tensor = torch.zeros(1).to(inputs), torch.zeros(args.n_classes).to(inputs)
        logpz, delta_logp = torch.zeros(1).to(inputs), torch.zeros(1).to(inputs)
        # z, _ = self.forward(inputs)
        # x = self.inverse(z)
        if args.dataset == 'celeba_5bit':
            nvals = 32
        elif args.dataset == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256

        # z, _ = self.forward(inputs)
        # x, _ = self.inverse(z)
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
        return bits_per_dim, logits_tensor, logpz, delta_logp, _

    def sample(self, batchSize):
        # TODO: Update this method for edge_index
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x

    def check_equivariance(self, r2_act, out_type, data, func, data_type=None):
        _, c, h, w = data.shape
        input_type = enn.FieldType(r2_act, self.c*[r2_act.trivial_repr])
        for g in r2_act.testing_elements:
            output = func(data)
            rg_output = enn.GeometricTensor(output.tensor.view(-1, c, h, w).cpu(),
                                         out_type).transform(g)
            data = enn.GeometricTensor(data.tensor.view(-1, c, h, w).cpu(), input_type)
            x_transformed = enn.GeometricTensor(data.transform(g).tensor.view(-1, c, h, w).cuda(), input_type)

            output_rg = func(x_transformed)
            # Equivariance Condition
            output_rg = enn.GeometricTensor(output_rg.tensor.cpu(), out_type)
            data = enn.GeometricTensor(data.tensor.squeeze().view(-1, c, h , w).cuda(), input_type)
            assert torch.allclose(rg_output.tensor.cpu().squeeze(), output_rg.tensor.squeeze(), atol=1e-5), g
        print("Passed Equivariance Test")

class EquivariantConvExp(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden,
                 group_action_type=None):
        super(EquivariantConvExp, self).__init__()
        mask = torch.arange(input_size).float() % 2
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden
        self.group_action_type = GROUPS[args.group]
        self.group_card = len(list(self.group_action_type.testing_elements))
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.p_z = StandardNormal
        self.input_type = enn.FieldType(self.group_action_type, [self.group_action_type.trivial_repr])
        self.flow_model = create_equivariant_convexp_blocks(input_size, hidden_size,
                                                   n_blocks, n_hidden,
                                                   self.group_action_type)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        self.mask = nn.Parameter(mask, requires_grad=False)

    def standard_normal_sample(self, size):
        return torch.randn(size)

    def standard_normal_logprob(self, z):
        logZ = -0.5 * math.log(2 * math.pi)
        return logZ - z.pow(2) / 2

    def inverse(self, z):
        log_det_J, x = z.new_zeros(z.shape[0]), z.view(-1, 1, 1, 2)
        B, C, H, W = x.size()
        x = enn.GeometricTensor(x, self.input_type)
        for i in range(0,self.n_blocks):
            x = self.flow_model[i](x, True)
            filter = self.flow_model[i][0].expand_parameters()[0]
            # filter = list(list(self.flow_model[i].modules())[1].modules())[1].expand_parameters()[0]
            log_det_J += log_det(filter) * H * W
        return x.tensor.squeeze(), log_det_J

    def forward(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x.view(-1, 1, 1, 2)
        B, C, H, W = z.size()
        z = enn.GeometricTensor(z, self.input_type)
        for i in reversed(range(0,self.n_blocks)):
            z = self.flow_model[i](z)
            # utils.check_equivariance(self.group_action_type, self.input_type,
                                     # x, self.flow_model[i], 'GeomTensor')
            filter = self.flow_model[i][0].expand_parameters()[0]
            # filter = list(list(self.flow_model[i].modules())[1].modules())[1].expand_parameters()[0]
            log_det_J -= log_det(filter) * H * W
        return z.tensor.squeeze(), log_det_J

    def log_prob(self, inputs, edge_index=None):
        z, delta_logp = self.forward(inputs)
        # compute log p(z)
        logpz = self.standard_normal_logprob(z).sum(1, keepdim=True)

        logpx = logpz - self.beta * delta_logp
        loss = -torch.mean(logpx)
        return loss, torch.mean(logpz), torch.mean(-delta_logp)

    def sample(self, batchSize):
        # TODO: Update this method for edge_index
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.inverse(z)
        return x


class EquivariantToyResFlow(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden,
                 group_action_type=None):
        super(EquivariantToyResFlow, self).__init__()
        self.args = args
        self.beta = args.beta
        self.n_blocks = n_blocks
        self.activation_fn = ACT_FNS[args.act]
        self.group_action_type = GROUPS[args.group]
        # self.group_action_type = gspaces.FlipRot2dOnR2(N=4)
        self.group_card = len(list(self.group_action_type.testing_elements))
        self.input_type = enn.FieldType(self.group_action_type, [self.group_action_type.trivial_repr])
        dims = [2] + list(map(int, args.dims.split('-'))) + [2]
        blocks = []
        if self.args.actnorm: blocks.append(layers.ActNorm1d(2))
        for _ in range(n_blocks):
            blocks.append(
                layers.Equivar_iResBlock(
                    self.build_nnet(dims, self.activation_fn),
                    n_dist=self.args.n_dist,
                    n_power_series=self.args.n_power_series,
                    exact_trace=self.args.exact_trace,
                    brute_force=self.args.brute_force,
                    n_samples=self.args.nsamples,
                    neumann_grad=True,
                    grad_in_forward=True,
                )
            )
            if self.args.actnorm: blocks.append(layers.ActNorm1d(2))
            if self.args.batchnorm: blocks.append(layers.MovingBatchNorm1d(2))
        self.flow_model = layers.SequentialFlow(blocks)

    def build_nnet(self, dims, activation_fn=enn.ReLU):
        nnet = []
        domains, codomains = self.parse_vnorms()
        if self.args.learn_p:
            if self.args.mixed:
                domains = [torch.nn.Parameter(torch.tensor(0.)) for _ in domains]
            else:
                domains = [torch.nn.Parameter(torch.tensor(0.))] * len(domains)
            codomains = domains[1:] + [domains[0]]

        in_type = enn.FieldType(self.group_action_type, [self.group_action_type.trivial_repr])
        out_dims = int(dims[1:][0] / self.group_card)
        out_type = enn.FieldType(self.group_action_type, out_dims*[self.group_action_type.regular_repr])
        total_layers = len(domains)
        for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
            nnet.append(
                base_layers.get_equivar_conv2d(
                    in_type,
                    out_type,
                    self.group_action_type,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    coeff=self.args.coeff,
                    n_iterations=self.args.n_lipschitz_iters,
                    atol=self.args.atol,
                    rtol=self.args.rtol,
                    domain=domain,
                    codomain=codomain,
                    zero_init=(out_dim == 2),
                )
            )
            nnet.append(activation_fn(nnet[-1].out_type, inplace=True))
            in_type = nnet[-1].out_type
            if i == total_layers - 2:
                out_type = enn.FieldType(self.group_action_type, [self.group_action_type.trivial_repr])
            else:
                out_type = enn.FieldType(self.group_action_type, out_dim*[self.group_action_type.regular_repr])

        return torch.nn.Sequential(*nnet)

    def update_lipschitz(self, n_iterations):
        for m in self.flow_model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True, n_iterations=n_iterations)
            if isinstance(m, base_layers.InducedNormEquivarConv2d):
                m.compute_weight(update=True, n_iterations=n_iterations)

    def compute_p_grads(self):
        scales = 0.
        nlayers = 0
        for m in self.flow_model.modules():
            if isinstance(m, base_layers.InducedNormEquivarConv2d):
                scales = scales + m.compute_one_iter()
                nlayers += 1
        scales.mul(1 / nlayers).mul(0.01).backward()
        for m in model.modules():
            if isinstance(m, base_layers.InducedNormEquivarConv2d):
                if m.domain.grad is not None and torch.isnan(m.domain.grad):
                    m.domain.grad = None

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def standard_normal_sample(self, size):
        return torch.randn(size)

    def standard_normal_logprob(self, z):
        logZ = -0.5 * math.log(2 * math.pi)
        return logZ - z.pow(2) / 2


    def parse_vnorms(self):
        ps = []
        for p in self.args.vnorms:
            if p == 'f':
                ps.append(float('inf'))
            else:
                ps.append(float(p))
        return ps[:-1], ps[1:]

    def inverse(self):
        return self.flow_model.inverse
        # return x

    def forward(self, x):
        x = enn.GeometricTensor(x.view(-1, 1, 1, 2), self.input_type)
        zero = torch.zeros(x.shape[0], 1).to(self.args.dev)

        # transform to z
        z, delta_logp = self.flow_model(x, zero)
        return z.tensor.squeeze(), delta_logp

    def log_prob(self, inputs):
        z, delta_logp = self.forward(inputs)
        # compute log p(z)
        logpz = self.standard_normal_logprob(z).sum(1, keepdim=True)

        logpx = logpz - self.beta * delta_logp
        loss = -torch.mean(logpx)
        return loss, torch.mean(logpz), torch.mean(-delta_logp)

    def sample(self, batchSize):
        pass

    def check_equivariance(self, r2_act, out_type, data, func, data_type=None):
        input_type = enn.FieldType(r2_act, [r2_act.trivial_repr])
        if data_type == 'GeomTensor':
            data = enn.GeometricTensor(data.view(-1, 1, 1, 2).cuda(), input_type)
        for g in r2_act.testing_elements:
            output = func(data)[0]
            rg_output = enn.GeometricTensor(output.tensor.view(-1,1,1,2).cpu(),
                                         out_type).transform(g)
            data = enn.GeometricTensor(data.tensor.view(-1, 1, 1, 2).cpu(), input_type)
            x_transformed = enn.GeometricTensor(data.transform(g).tensor.view(-1,1,1,2).cuda(),
                                input_type)

            output_rg = func(x_transformed)[0]
            # Equivariance Condition
            output_rg = enn.GeometricTensor(output_rg.tensor.cpu(), out_type)
            data = enn.GeometricTensor(data.tensor.squeeze().view(-1,1,1,2).cuda(),input_type)
            assert torch.allclose(rg_output.tensor.cpu().squeeze(), output_rg.tensor.squeeze(), atol=1e-5), g
