import numpy as np
import torch
import torch.nn as nn

import math
import flows.layers.base as base_layers
import flows.layers as layers
import ipdb

ACT_FNS = {
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

class ToyResFlow(nn.Module):
    def __init__(self, args, n_blocks, input_size, hidden_size, n_hidden):
        super(ToyResFlow, self).__init__()
        self.args = args
        self.beta = args.beta
        self.n_blocks = n_blocks
        self.input_size = input_size
        self.activation_fn = ACT_FNS[args.act]
        dims = [args.input_dim] + list(map(int, args.dims.split('-'))) + [args.input_dim]
        blocks = []
        if self.args.actnorm: blocks.append(layers.ActNorm1d(2))
        for _ in range(n_blocks):
            blocks.append(
                layers.iResBlock(
                    self.build_nnet(dims, self.activation_fn),
                    n_dist=self.args.n_dist,
                    n_power_series=self.args.n_power_series,
                    exact_trace=self.args.exact_trace,
                    brute_force=self.args.brute_force,
                    n_samples=self.args.batch_size,
                    neumann_grad=True,
                    grad_in_forward=True,
                )
            )
            if self.args.actnorm: blocks.append(layers.ActNorm1d(args.input_dim))
            if self.args.batchnorm:
                blocks.append(layers.MovingBatchNorm1d(args.input_dim))
        self.flow_model = layers.SequentialFlow(blocks)

    def build_nnet(self, dims, activation_fn=torch.nn.ReLU):
        nnet = []
        domains, codomains = self.parse_vnorms()
        if self.args.learn_p:
            if self.args.mixed:
                domains = [torch.nn.Parameter(torch.tensor(0.)) for _ in domains]
            else:
                domains = [torch.nn.Parameter(torch.tensor(0.))] * len(domains)
            codomains = domains[1:] + [domains[0]]
        for i, (in_dim, out_dim, domain, codomain) in enumerate(zip(dims[:-1], dims[1:], domains, codomains)):
            nnet.append(activation_fn())
            nnet.append(
                base_layers.get_linear(
                    in_dim,
                    out_dim,
                    coeff=self.args.coeff,
                    n_iterations=self.args.n_lipschitz_iters,
                    atol=self.args.atol,
                    rtol=self.args.rtol,
                    domain=domain,
                    codomain=codomain,
                    zero_init=(out_dim == 2),
                )
            )

        return torch.nn.Sequential(*nnet)

    def update_lipschitz(self, n_iterations):
        for m in self.flow_model.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                m.compute_weight(update=True, n_iterations=n_iterations)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                m.compute_weight(update=True, n_iterations=n_iterations)

    def compute_p_grads(self):
        scales = 0.
        nlayers = 0
        for m in self.flow_model.modules():
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                scales = scales + m.compute_one_iter()
                nlayers += 1
        scales.mul(1 / nlayers).mul(0.01).backward()
        for m in model.modules():
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
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
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to z
        z, delta_logp = self.flow_model(x, zero)
        return z, delta_logp

    def log_prob(self, inputs, prior=None):
        z, delta_logp = self.forward(inputs)
        # compute log p(z)
        if prior is None:
            logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
            # logpz = self.standard_normal_logprob(z).sum(1, keepdim=True)
        else:
            # ipdb.set_trace()
            # z = z.view(-1, inputs.shape[1])
            logpz = -1*prior.energy(z)
            # nll = (prior.energy(z).view(-1) + delta_logp.view(-1)).mean()

        logpx = logpz - self.beta * delta_logp
        loss = -torch.mean(logpx)
        return loss, torch.mean(logpz), torch.mean(-delta_logp)
        # logpx = logpz - self.beta * delta_logp
        # return logpx

    def sample(self, batchSize):
        pass
