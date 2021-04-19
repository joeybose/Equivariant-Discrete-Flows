import torch
import torch.nn as nn
import os
import os.path as osp
import argparse
from torchvision.utils import save_image

from collections import OrderedDict
import torch
import math
from statistics import median, mean
import random
import numpy as np
import copy
import logging
from torch._six import inf
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import flows.layers.base as base_layers
import flows.layers as layers
from e2cnn import gspaces
from e2cnn import nn as enn
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

LOW = -4
HIGH = 4

def check_equivariance(r2_act, out_type, data, func, data_type=None):
    input_type = enn.FieldType(r2_act, [r2_act.trivial_repr])
    if data_type == 'GeomTensor':
        data = enn.GeometricTensor(data.view(-1, 1, 1, 2), input_type)
    for g in r2_act.testing_elements:
        output = func(data)
        if data_type == 'GeomTensor':
            rg_output = enn.GeometricTensor(output.tensor.view(-1,1,1,2).cpu(),
                                         out_type).transform(g)
            data = enn.GeometricTensor(data.tensor.view(-1, 1, 1, 2).cpu(), input_type)
            x_transformed = enn.GeometricTensor(data.transform(g).tensor.cuda().view(-1,1,1,2),
                                                input_type)
        else:
            rg_output = enn.GeometricTensor(output.view(-1,1,1,2).cpu(),
                                            out_type).transform(g)
            data = enn.GeometricTensor(data.view(-1, 1, 1, 2).cpu(), input_type)
            x_transformed = data.transform(g).tensor.cuda().view(-1,1,1,2)

        output_rg = func(x_transformed)
        # Equivariance Condition
        if data_type == 'GeomTensor':
            output_rg = enn.GeometricTensor(output_rg.tensor.cpu(), out_type)
            data = enn.GeometricTensor(data.tensor.squeeze().cuda().view(-1,1,1,2),input_type)
        else:
            output_rg = enn.GeometricTensor(output_rg.view(-1,1,1,2).cpu(), out_type)
            data = data.tensor.squeeze().cuda()
        assert torch.allclose(rg_output.tensor.cpu().squeeze(), output_rg.tensor.squeeze(), atol=1e-5), g

def check_invariance(r2_act, out_type, data, func):
    input_type = enn.FieldType(r2_act, [r2_act.trivial_repr])
    data = enn.GeometricTensor(data.view(-1, 1, 1, 2), input_type)
    for g in r2_act.testing_elements:
        log_prob = func(data)
        data = enn.GeometricTensor(data.tensor.view(-1, 1, 1, 2).cpu(), input_type)
        x_transformed = enn.GeometricTensor(data.transform(g).tensor.cuda().view(-1,1,1,2),
                                            input_type)
        invar_new_log_prob = func(x_transformed)
        data = enn.GeometricTensor(data.tensor.squeeze().cuda().view(-1,1,1,2),input_type)
        assert torch.allclose(log_prob.tensor.cpu().squeeze(),
                              equivar_log_prob.tensor.squeeze(), atol=1e-5), g

class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        ipdb.set_trace()
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

def create_selfloop_edges(num_nodes):
    edges = []
    for i in range(0, num_nodes):
        edges.append((int(i),int(i)))

    return edges

def perm_node_feats(feats):
    num_nodes = feats.size(0)
    perm = torch.randperm(feats.size(0))
    perm_idx = perm[:num_nodes]
    feats = feats[perm_idx]
    return feats

def log_mean_exp(value, dim=0, keepdim=False):
    return log_sum_exp(value, dim, keepdim) - math.log(value.size(dim))


def log_sum_exp(value, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))


def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))

def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")

def filter_state_dict(state_dict,name):
    keys_to_del = []
    for key in state_dict.keys():
        if name in key:
            keys_to_del.append(key)
    for key in sorted(keys_to_del, reverse=True):
        del state_dict[key]
    return state_dict

''' Set Random Seed '''
def seed_everything(seed):
    if seed:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def custom_clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p is not int(0), parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.data.mul_(clip_coef)
    return total_norm

'''Monitor Norm of gradients'''
def monitor_grad_norm(model):
    parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

'''Monitor Norm of gradients'''
def monitor_grad_norm_2(gradients):
    total_norm = 0
    for p in gradients:
        if p is not int(0):
            param_norm = p.data.norm(2)
            total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

'''Monitor Norm of weights'''
def monitor_weight_norm(model):
    parameters = list(filter(lambda p: p is not None, model.parameters()))
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=20, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def project_name(dataset_name):
    if dataset_name:
        return "Equivar-Flows-{}".format(dataset_name)
    else:
        return "Equivar-Flows"


class Constants(object):
    eta = 1e-5
    log2 = math.log(2)
    logpi = math.log(math.pi)
    log2pi = math.log(2 * math.pi)
    logceilc = 88                # largest cuda v s.t. exp(v) < inf
    logfloorc = -104             # smallest cuda v s.t. exp(v) > 0
    invsqrt2pi = 1. / math.sqrt(2 * math.pi)
    sqrthalfpi = math.sqrt(math.pi/2)

def plt_potential_func(potential, ax, npts=100, title="$p(x)$"):
    """
    Args:
        potential: computes U(z_k) given z_k
    """
    xside = np.linspace(LOW, HIGH, npts)
    yside = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = potential(z).cpu().numpy()
    p = np.exp(-u).reshape(npts, npts)

    plt.pcolormesh(xx, yy, p)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow(prior_logdensity, transform, ax, npts=100, title="$q(x)$", device="cpu"):
    """
    Args:
        transform: computes z_k and log(q_k) given z_0
    """
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.tensor(z, requires_grad=True).type(torch.float32).to(device)
    logqz = prior_logdensity(z)
    logqz = torch.sum(logqz, dim=1)[:, None]
    z, logqz = transform(z, logqz)
    logqz = torch.sum(logqz, dim=1)[:, None]

    xx = z[:, 0].cpu().numpy().reshape(npts, npts)
    yy = z[:, 1].cpu().numpy().reshape(npts, npts)
    qz = np.exp(logqz.cpu().numpy()).reshape(npts, npts)

    plt.pcolormesh(xx, yy, qz)
    ax.set_xlim(LOW, HIGH)
    ax.set_ylim(LOW, HIGH)
    cmap = matplotlib.cm.get_cmap(None)
    ax.set_facecolor(cmap(0.))
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_density(prior_logdensity, inverse_transform, ax, npts=100, memory=100, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    zeros = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = [], []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        z_, delta_logp_ = inverse_transform(x[ii], zeros[ii])
        if not torch.is_tensor(z_):
            z_ = z_.tensor.squeeze()
        z.append(z_)
        delta_logp.append(delta_logp_)
    z = torch.cat(z, 0)
    delta_logp = torch.cat(delta_logp, 0)

    logpz = prior_logdensity(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)

    ax.imshow(px, cmap=cm.magma)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_samples(prior_sample, transform, ax, npts=100, memory=100, title="$x ~ q(x)$", device="cpu"):
    z = prior_sample(npts * npts, 2).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        zk.append(transform(z[ii]))
    zk = torch.cat(zk, 0).squeeze().cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts, cmap=cm.magma)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_samples(samples, ax, npts=100, title="$x ~ p(x)$"):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts, cmap=cm.magma)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def visualize(args, epoch, flow_model, itr, real_imgs):
    flow_model.eval()
    makedirs(os.path.join(args.save, 'imgs'))
    real_imgs = real_imgs[:32]
    _real_imgs = real_imgs

    fixed_z = torch.randn([min(32, args.batch_size), (args.im_dim +
                                                          args.padding)
                                *args.imagesize * args.imagesize]).to(args.dev)
    if args.dataset == 'celeba_5bit':
        nvals = 32
    elif args.dataset == 'celebahq':
        nvals = 2**args.nbits
    else:
        nvals = 256

    with torch.no_grad():
        # reconstructed real images
        real_imgs, _ = add_padding(args, real_imgs, nvals)
        if args.squeeze_first: real_imgs = squeeze_layer(real_imgs)
        recon_imgs = flow_model(real_imgs.view(-1, *args.input_size[1:]),
                                inverse=True).view(-1, *args.input_size[1:])
        if args.squeeze_first: recon_imgs = squeeze_layer.inverse(recon_imgs)
        recon_imgs = remove_padding(args, recon_imgs)

        # random samples
        fake_imgs = flow_model(fixed_z, inverse=True).view(-1, *args.input_size[1:])
        if args.squeeze_first: fake_imgs = squeeze_layer.inverse(fake_imgs)
        fake_imgs = remove_padding(args, fake_imgs)

        fake_imgs = fake_imgs.view(-1, args.im_dim, args.imagesize, args.imagesize)
        recon_imgs = recon_imgs.view(-1, args.im_dim, args.imagesize, args.imagesize)
        imgs = torch.cat([_real_imgs, fake_imgs, recon_imgs], 0)

        filename = os.path.join(args.save, 'imgs', 'e{:03d}_i{:06d}.png'.format(epoch, itr))
        save_image(imgs.cpu().float(), filename, nrow=16, padding=2)
    flow_model.train()

def visualize_transform(
    potential_or_samples, prior_sample, prior_density, transform=None, inverse_transform=None, samples=False, npts=100,
    memory=100, device="cpu"
):
    """Produces visualization for the model density and samples from the model."""
    plt.clf()
    ax = plt.subplot(1, 3, 1, aspect="equal")
    if samples:
        plt_samples(potential_or_samples, ax, npts=npts)
    else:
        plt_potential_func(potential_or_samples, ax, npts=npts)

    ax = plt.subplot(1, 3, 2, aspect="equal")
    if inverse_transform is None:
        plt_flow(prior_density, transform, ax, npts=npts, device=device)
    else:
        plt_flow_density(prior_density, inverse_transform, ax, npts=npts, memory=memory, device=device)

    ax = plt.subplot(1, 3, 3, aspect="equal")
    if transform is not None:
        plt_flow_samples(prior_sample, transform, ax, npts=npts, memory=memory, device=device)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

class ExponentialMovingAverage(object):

    def __init__(self, module, decay=0.999):
        """Initializes the model when .apply() is called the first time.
        This is to take into account data-dependent initialization that occurs in the first iteration."""
        self.module = module
        self.decay = decay
        self.shadow_params = {}
        self.nparams = sum(p.numel() for p in module.parameters())

    def init(self):
        for name, param in self.module.named_parameters():
            self.shadow_params[name] = param.data.clone()

    def apply(self):
        if len(self.shadow_params) == 0:
            self.init()
        else:
            with torch.no_grad():
                for name, param in self.module.named_parameters():
                    self.shadow_params[name] -= (1 - self.decay) * (self.shadow_params[name] - param.data)

    def set(self, other_ema):
        self.init()
        with torch.no_grad():
            for name, param in other_ema.shadow_params.items():
                self.shadow_params[name].copy_(param)

    def replace_with_ema(self):
        for name, param in self.module.named_parameters():
            param.data.copy_(self.shadow_params[name])

    def swap(self):
        for name, param in self.module.named_parameters():
            tmp = self.shadow_params[name].clone()
            self.shadow_params[name].copy_(param.data)
            param.data.copy_(tmp)

    def __repr__(self):
        return (
            '{}(decay={}, module={}, nparams={})'.format(
                self.__class__.__name__, self.decay, self.module.__class__.__name__, self.nparams
            )
        )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x


def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x


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


def remove_padding(args, x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'

def update_lr(args, optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class CosineAnnealingWarmRestarts(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{i}}\pi))
    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0`(after restart), set :math:`\eta_t=\eta_{max}`.
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations for the first restart.
        T_mult (int, optional): A factor increases :math:`T_{i}` after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of last epoch. Default: -1.
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mul >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)
        self.T_cur = last_epoch

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step could be called after every update, i.e. if one epoch has 10 iterations
        (number_of_train_examples / batch_size), we should call SGDR.step(0.1), SGDR.step(0.2), etc.
        This function can be called in an interleaved way.
        Example:
            >>> scheduler = SGDR(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult**(n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def estimator_moments(model, baseline=0):
    avg_first_moment = 0.
    avg_second_moment = 0.
    for m in model.modules():
        if isinstance(m, layers.iResBlock):
            avg_first_moment += m.last_firmom.item()
            avg_second_moment += m.last_secmom.item()
    return avg_first_moment, avg_second_moment

def parallelize(model):
    return torch.nn.DataParallel(model)
