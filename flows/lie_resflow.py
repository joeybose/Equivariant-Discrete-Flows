import numpy as np
import torch
import torch.nn as nn
import math
import flows.layers.base as base_layers
import flows.layers as layers
import numpy as np
from flows.layers.base.lie_conv.lieGroups import T,SO2,SO3,SE2,SE3, norm
import ipdb
from flows.distributions import HypersphericalUniform

ACT_FNS = {
    'softplus': lambda b: nn.Softplus(),
    'elu': lambda b: nn.ELU(inplace=b),
    'swish': lambda b: base_layers.Swish(),
    'lcube': lambda b: base_layers.LipschitzCube(),
    'identity': lambda b: base_layers.Identity(),
    'relu': lambda b: nn.ReLU(inplace=b),
}

GROUPS = {
    'so2': SO2,
    'so3': SO3,
    'se2': SE2,
    'se3': SE3,
    't': T,
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


class LieResidualFlow(nn.Module):

    def __init__(
        self,
        args,
        input_size,
        n_blocks=[16, 16],
        intermediate_dim=64,
        factor_out=True,
        quadratic=False,
        init_layer=None,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
        classification=False,
        classification_hdim=64,
        n_classes=10,
        block_type='resblock',
    ):
        super(LieResidualFlow, self).__init__()
        self.n_scale = min(len(n_blocks), self._calc_n_scale(input_size))
        _, self.c, self.h, self.w = input_size[:]
        self.n_blocks = n_blocks
        self.intermediate_dim = intermediate_dim
        self.factor_out = factor_out
        self.quadratic = quadratic
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_actnorm = fc_actnorm
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.fc = fc
        self.coeff = coeff
        self.vnorms = vnorms
        self.n_lipschitz_iters = n_lipschitz_iters
        self.sn_atol = sn_atol
        self.sn_rtol = sn_rtol
        self.n_power_series = n_power_series
        self.n_dist = n_dist
        self.n_samples = n_samples
        self.kernels = kernels
        self.activation_fn = activation_fn
        self.fc_end = fc_end
        self.fc_idim = fc_idim
        self.n_exact_terms = n_exact_terms
        self.preact = preact
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.first_resblock = first_resblock
        self.learn_p = learn_p
        self.classification = classification
        self.classification_hdim = classification_hdim
        self.n_classes = n_classes
        self.block_type = block_type

        self.group = GROUPS[args.group]()
        self.nbhd = 25
        self.ds_frac = 1
        self.fill = 0.1
        self.bn = True
        self.mean = True
        self.liftsamples = 1
        self.prior = HypersphericalUniform(dim=self.c*self.h*self.w,
                                           device=args.dev)
        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)
        ipdb.set_trace()
        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

        if self.classification:
            self.build_multiscale_classifier(input_size)

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        _stacked_blocks = StackediResBlocks if self.block_type == 'resblock' else StackedCouplingBlocks
        for i in range(self.n_scale):
            transforms.append(
                _stacked_blocks(
                    self.group,
                    initial_size=(c, h, w),
                    idim=self.intermediate_dim,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None,
                    n_blocks=self.n_blocks[i],
                    quadratic=self.quadratic,
                    actnorm=self.actnorm,
                    fc_actnorm=self.fc_actnorm,
                    batchnorm=self.batchnorm,
                    dropout=self.dropout,
                    fc=self.fc,
                    coeff=self.coeff,
                    vnorms=self.vnorms,
                    n_lipschitz_iters=self.n_lipschitz_iters,
                    sn_atol=self.sn_atol,
                    sn_rtol=self.sn_rtol,
                    n_power_series=self.n_power_series,
                    n_dist=self.n_dist,
                    n_samples=self.n_samples,
                    kernels=self.kernels,
                    activation_fn=self.activation_fn,
                    fc_end=self.fc_end,
                    fc_idim=self.fc_idim,
                    n_exact_terms=self.n_exact_terms,
                    preact=self.preact,
                    neumann_grad=self.neumann_grad,
                    grad_in_forward=self.grad_in_forward,
                    first_resblock=self.first_resblock and (i == 0),
                    learn_p=self.learn_p,
                    nbhd=self.nbhd,
                    ds_frac=self.ds_frac,
                    fill=self.fill,
                    bn=self.bn,
                    mean=self.mean,
                )
            )
            c, h, w = c * 2 if self.factor_out else c * 4, h // 2, w // 2
        return nn.ModuleList(transforms)

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 4**k, h // 2**k, w // 2**k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def build_multiscale_classifier(self, input_size):
        n, c, h, w = input_size
        hidden_shapes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2 if self.factor_out else 4
                h //= 2
                w //= 2
            hidden_shapes.append((n, c, h, w))

        classification_heads = []
        for i, hshape in enumerate(hidden_shapes):
            classification_heads.append(
                nn.Sequential(
                    nn.Conv2d(hshape[1], self.classification_hdim, 3, 1, 1),
                    layers.ActNorm2d(self.classification_hdim),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
            )
        self.classification_heads = nn.ModuleList(classification_heads)
        self.logit_layer = nn.Linear(self.classification_hdim * len(classification_heads), self.n_classes)

    def forward(self, x, logpx=None, inverse=False, classify=False,
                coord_transform=None):

        """ assumes x is a regular image: (bs,c,h,w)"""
        ipdb.set_trace()
        bs,c,h,w = x.shape
        # Construct coordinate grid
        i = torch.linspace(-h/2.,h/2.,h)
        j = torch.linspace(-w/2.,w/2.,w)
        coords = torch.stack(torch.meshgrid([i,j]),dim=-1).float()
        # Perform center crop
        # center_mask = coords.norm(dim=-1)<15. # crop out corners (filled only with zeros)
        center_mask = coords.norm(dim=-1)<100. # crop out corners (filled only with zeros)
        coords = coords[center_mask].view(-1,2).unsqueeze(0).repeat(bs,1,1).to(x.device)
        if coord_transform is not None: coords = coord_transform(coords)
        values = x.permute(0,2,3,1)[:,center_mask,:].reshape(bs,-1,c)
        mask = torch.ones(bs,values.shape[1],device=x.device)>0 # all true
        z = (coords,values,mask)
        # Perform lifting of the coordinates and cache results
        with torch.no_grad():
            lifted_coords,lifted_vals,lifted_mask = self.group.lift(z, self.liftsamples)

        tuple_x =  (lifted_coords,lifted_vals,lifted_mask)
        if inverse:
            return self.inverse(tuple_x, logpx)

        out = []
        if classify: class_outs = []
        for idx in range(len(self.transforms)):
            if logpx is not None:
                tuple_x, logpx = self.transforms[idx].forward(tuple_x, logpx)
            else:
                tuple_x = self.transforms[idx].forward(tuple_x)
            if self.factor_out and (idx < len(self.transforms) - 1):
                d = tuple_x[1].size(1) // 2
                tuple_x[1], f = tuple_x[1][:, :d], tuple_x[1][:, d:]
                out.append(f)

            # Handle classification.
            if classify:
                if self.factor_out:
                    class_outs.append(self.classification_heads[idx](f))
                else:
                    class_outs.append(self.classification_heads[idx](tuple_x))

        lifted_coords, x,lifted_mask = tuple_x[:]
        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)
        output = out if logpx is None else (out, logpx)
        if classify:
            h = torch.cat(class_outs, dim=1).squeeze(-1).squeeze(-1)
            logits = self.logit_layer(h)
            return output, logits
        else:
            return output

    def inverse(self, z, logpz=None):
        if self.factor_out:
            z = z.view(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, i:i + s])
                i += s
            zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].inverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev = self.transforms[idx].inverse(z_prev)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].inverse(zs[-1], logpz)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev, logpz = self.transforms[idx].inverse(z_prev, logpz)
                return z_prev, logpz
        else:
            z = z.view(z.shape[0], *self.dims[-1])
            for idx in range(len(self.transforms) - 1, -1, -1):
                if logpz is None:
                    z = self.transforms[idx].inverse(z)
                else:
                    z, logpz = self.transforms[idx].inverse(z, logpz)
            return z if logpz is None else (z, logpz)

    def check_invertibility(self, args, x):
        if args.dataset == 'celeba_5bit':
            nvals = 32
        elif args.dataset == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256
        x, logpu = add_padding(args, x, nvals)
        z, delta_logp = self.forward(x.view(-1, *args.input_size[1:]), 0)
        inv = self.forward(z.view(-1, *args.input_size[1:]), inverse=True)

        atol_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        batch_size = x.shape[0]
        diff = x.view(batch_size, -1) - inv.view(batch_size, -1)
        avg_norm_diff = torch.norm(diff, p='fro', dim=-1).mean()
        print("Avg Diff is %f" %(avg_norm_diff))
        for atol in atol_list:
            res = torch.allclose(x, inv, atol)
            print("Invertiblity at %f: %s" %(atol, str(res)))
        return avg_norm_diff

    def compute_avg_test_loss(self, args, r2_act, data, beta=1.):
        _, c, h, w = data.shape
        input_type = enn.FieldType(r2_act, self.c*[r2_act.trivial_repr])
        bits_per_dim, logits_tensor = torch.zeros(1).to(data), torch.zeros(args.n_classes).to(data)
        logpz, delta_logp = torch.zeros(1).to(data), torch.zeros(1).to(data)
        logpx_list = []
        data = enn.GeometricTensor(data.cpu(), self.input_type)
        if args.dataset == 'celeba_5bit':
            nvals = 32
        elif args.dataset == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256
        for g in r2_act.testing_elements:
            x_transformed = data.transform(g).tensor.view(-1, c, h, w).cuda()
            padded_inputs, logpu = add_padding(args, x_transformed, nvals)
            z, delta_logp = self.forward(padded_inputs.view(-1, *args.input_size[1:]), 0)
            logpz = self.prior.log_prob(z).view(z.size(0), -1).sum(1, keepdim=True)
            # logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

            # log p(x)
            logpx = logpz - beta * delta_logp - np.log(nvals) * (
                args.imagesize * args.imagesize * (args.im_dim + args.padding)
            ) - logpu
            logpx_list.append(logpx)

        logpx_total = torch.vstack(logpx_list)
        bits_per_dim = -torch.mean(logpx_total) / (args.imagesize *
                                             args.imagesize * args.im_dim) / np.log(2)
        return bits_per_dim

    def compute_loss(self, args, x, beta=1.0, do_test=False):
        # if do_test and not args.task == 'hybrid':
            # # ipdb.set_trace()
            # return self.compute_avg_test_loss(args, self.group_action_type, x)
        bits_per_dim, logits_tensor = torch.zeros(1).to(x), torch.zeros(args.n_classes).to(x)
        logpz, delta_logp = torch.zeros(1).to(x), torch.zeros(1).to(x)

        if args.dataset == 'celeba_5bit':
            nvals = 32
        elif args.dataset == 'celebahq':
            nvals = 2**args.nbits
        else:
            nvals = 256

        x, logpu = add_padding(args, x, nvals)

        if args.squeeze_first:
            x = squeeze_layer(x)

        if args.task == 'hybrid':
            z_logp, logits_tensor = self.forward(x.view(-1, *args.input_size[1:]), 0, classify=True)
            z, delta_logp = z_logp
        elif args.task == 'density':
            z, delta_logp = self.forward(x.view(-1, *args.input_size[1:]), 0)
        elif args.task == 'classification':
            z, logits_tensor = self.forward(x.view(-1, *args.input_size[1:]), classify=True)

        if args.task in ['density', 'hybrid']:
            # log p(z)
            logpz = self.prior.log_prob(z).view(z.size(0), -1).sum(1, keepdim=True)
            # logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)

            # log p(x)
            logpx = logpz - beta * delta_logp - np.log(nvals) * (
                args.imagesize * args.imagesize * (args.im_dim + args.padding)
            ) - logpu
            bits_per_dim = -torch.mean(logpx) / (args.imagesize *
                                                 args.imagesize * args.im_dim) / np.log(2)

            logpz = torch.mean(logpz).detach()
            delta_logp = torch.mean(-delta_logp).detach()

        return bits_per_dim, logits_tensor, logpz, delta_logp, z

    def update_lipschitz(self, n_iterations):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                    m.compute_weight(update=True, n_iterations=n_iterations)
                if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                    m.compute_weight(update=True, n_iterations=n_iterations)

    def get_lipschitz_constants(self):
        lipschitz_constants = []
        for m in self.modules():
            if isinstance(m, base_layers.SpectralNormConv2d) or isinstance(m, base_layers.SpectralNormLinear):
                lipschitz_constants.append(m.scale)
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                lipschitz_constants.append(m.scale)
            if isinstance(m, base_layers.LopConv2d) or isinstance(m, base_layers.LopLinear):
                lipschitz_constants.append(m.scale)
        return lipschitz_constants

    def compute_p_grads(self):
        scales = 0.
        nlayers = 0
        for m in self.modules():
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                scales = scales + m.compute_one_iter()
                nlayers += 1
        scales.mul(1 / nlayers).backward()
        for m in self.modules():
            if isinstance(m, base_layers.InducedNormConv2d) or isinstance(m, base_layers.InducedNormLinear):
                if m.domain.grad is not None and torch.isnan(m.domain.grad):
                    m.domain.grad = None

class StackediResBlocks(layers.SequentialFlow):

    def __init__(
        self,
        group,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
        nbhd=25,
        ds_frac=1.0,
        fill=0.1,
        bn=True,
        mean=True,
    ):

        chain = []
        self.nbhd = nbhd
        self.ds_frac = ds_frac
        self.fill = fill
        self.bn = bn
        self.mean = mean
        self.group = group
        # Parse vnorms
        ps = []
        for p in vnorms:
            if p == 'f':
                ps.append(float('inf'))
            else:
                ps.append(float(p))
        domains, codomains = ps[:-1], ps[1:]
        assert len(domains) == len(kernels.split('-'))

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, h, w = initial_size
                dim = c * h * w
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv2d(initial_size[0])

        def _lipschitz_layer(fc):
            return base_layers.get_lie_conv2d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                return layers.iResBlock(
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_lipschitz_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        coeff=coeff,
                        domains=domains,
                        codomains=codomains,
                        n_iterations=n_lipschitz_iters,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        sn_atol=sn_atol,
                        sn_rtol=sn_rtol,
                        learn_p=learn_p,
                    ),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )
            else:
                ks = list(map(int, kernels.split('-')))
                if learn_p:
                    _domains = [nn.Parameter(torch.tensor(0.)) for _ in range(len(ks))]
                    _codomains = _domains[1:] + [_domains[0]]
                else:
                    _domains = domains
                    _codomains = codomains
                nnet = []
                if not first_resblock and preact:
                    if batchnorm: nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(
                    _lipschitz_layer(fc)( initial_size[0], idim, self.nbhd,
                                         self.ds_frac, self.bn, activation_fn,
                                         self.mean, self.group, self.fill,
                                         coeff=coeff,
                                         n_iterations=n_lipschitz_iters,
                                         domain=_domains[0],
                                         codomain=_codomains[0], atol=sn_atol,
                                         rtol=sn_rtol)
                )
                if batchnorm: nnet.append(layers.MovingBatchNorm2d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(
                        _lipschitz_layer(fc)( idim, idim, self.nbhd,
                                             self.ds_frac, self.bn,
                                             activation_fn, self.mean,
                                             self.group, self.fill,
                                             coeff=coeff,
                                             n_iterations=n_lipschitz_iters,
                                             domain=_domains[i+1],
                                             codomain=_codomains[i+1],
                                             atol=sn_atol, rtol=sn_rtol)
                    )
                    if batchnorm: nnet.append(layers.MovingBatchNorm2d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                if dropout: nnet.append(nn.Dropout2d(dropout, inplace=True))
                nnet.append(
                    _lipschitz_layer(fc)( idim, initial_size[0], self.nbhd,
                                         self.ds_frac, self.bn, activation_fn,
                                         self.mean, self.group, self.fill,
                                         coeff=coeff,
                                         n_iterations=n_lipschitz_iters,
                                         domain=_domains[-1],
                                         codomain=_codomains[-1], atol=sn_atol,
                                         rtol=sn_rtol)
                )
                if batchnorm: nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                return layers.iResBlock(
                    nn.Sequential(*nnet),
                    n_power_series=n_power_series,
                    n_dist=n_dist,
                    n_samples=n_samples,
                    n_exact_terms=n_exact_terms,
                    neumann_grad=neumann_grad,
                    grad_in_forward=grad_in_forward,
                )
        ipdb.set_trace()
        if init_layer is not None: chain.append(init_layer)
        if first_resblock and actnorm: chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm: chain.append(_actnorm(initial_size, True))
        if squeeze:
            c, h, w = initial_size
            for i in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (i == 0)))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            chain.append(layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm: chain.append(_actnorm(initial_size, True))

        super(StackediResBlocks, self).__init__(chain)


class FCNet(nn.Module):

    def __init__(
        self, input_shape, idim, lipschitz_layer, nhidden, coeff, domains, codomains, n_iterations, activation_fn,
        preact, dropout, sn_atol, sn_rtol, learn_p, div_in=1
    ):
        super(FCNet, self).__init__()
        self.input_shape = input_shape
        c, h, w = self.input_shape
        dim = c * h * w
        nnet = []
        last_dim = dim // div_in
        if preact: nnet.append(ACT_FNS[activation_fn](False))
        if learn_p:
            domains = [nn.Parameter(torch.tensor(0.)) for _ in range(len(domains))]
            codomains = domains[1:] + [domains[0]]
        for i in range(nhidden):
            nnet.append(
                lipschitz_layer(last_dim, idim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                    last_dim, idim, coeff=coeff, n_iterations=n_iterations, domain=domains[i], codomain=codomains[i],
                    atol=sn_atol, rtol=sn_rtol
                )
            )
            nnet.append(ACT_FNS[activation_fn](True))
            last_dim = idim
        if dropout: nnet.append(nn.Dropout(dropout, inplace=True))
        nnet.append(
            lipschitz_layer(last_dim, dim) if lipschitz_layer == nn.Linear else lipschitz_layer(
                last_dim, dim, coeff=coeff, n_iterations=n_iterations, domain=domains[-1], codomain=codomains[-1],
                atol=sn_atol, rtol=sn_rtol
            )
        )
        self.nnet = nn.Sequential(*nnet)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = self.nnet(x)
        return y.view(y.shape[0], *self.input_shape)


class FCWrapper(nn.Module):

    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward(self, x, logpx=None):
        shape = x.shape
        x = x.view(x.shape[0], -1)
        if logpx is None:
            y = self.fc_module(x)
            return y.view(*shape)
        else:
            y, logpy = self.fc_module(x, logpx)
            return y.view(*shape), logpy

    def inverse(self, y, logpy=None):
        shape = y.shape
        y = y.view(y.shape[0], -1)
        if logpy is None:
            x = self.fc_module.inverse(y)
            return x.view(*shape)
        else:
            x, logpx = self.fc_module.inverse(y, logpy)
            return x.view(*shape), logpx


class StackedCouplingBlocks(layers.SequentialFlow):

    def __init__(
        self,
        initial_size,
        idim,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        quadratic=False,
        actnorm=False,
        fc_actnorm=False,
        batchnorm=False,
        dropout=0,
        fc=False,
        coeff=0.9,
        vnorms='122f',
        n_lipschitz_iters=None,
        sn_atol=None,
        sn_rtol=None,
        n_power_series=5,
        n_dist='geometric',
        n_samples=1,
        kernels='3-1-3',
        activation_fn='elu',
        fc_end=True,
        fc_nblocks=4,
        fc_idim=128,
        n_exact_terms=0,
        preact=False,
        neumann_grad=True,
        grad_in_forward=False,
        first_resblock=False,
        learn_p=False,
    ):

        # yapf: disable
        class nonloc_scope: pass
        nonloc_scope.swap = True
        # yapf: enable

        chain = []

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(layers.ActNorm1d(size[0] * size[1] * size[2]))
            else:
                return layers.ActNorm2d(size[0])

        def _quadratic_layer(initial_size, fc):
            if fc:
                c, h, w = initial_size
                dim = c * h * w
                return FCWrapper(layers.InvertibleLinear(dim))
            else:
                return layers.InvertibleConv2d(initial_size[0])

        def _weight_layer(fc):
            return nn.Linear if fc else nn.Conv2d

        def _resblock(initial_size, fc, idim=idim, first_resblock=False):
            if fc:
                nonloc_scope.swap = not nonloc_scope.swap
                return layers.CouplingBlock(
                    initial_size[0],
                    FCNet(
                        input_shape=initial_size,
                        idim=idim,
                        lipschitz_layer=_weight_layer(True),
                        nhidden=len(kernels.split('-')) - 1,
                        activation_fn=activation_fn,
                        preact=preact,
                        dropout=dropout,
                        coeff=None,
                        domains=None,
                        codomains=None,
                        n_iterations=None,
                        sn_atol=None,
                        sn_rtol=None,
                        learn_p=None,
                        div_in=2,
                    ),
                    swap=nonloc_scope.swap,
                )
            else:
                ks = list(map(int, kernels.split('-')))

                if init_layer is None:
                    _block = layers.ChannelCouplingBlock
                    _mask_type = 'channel'
                    div_in = 2
                    mult_out = 1
                else:
                    _block = layers.MaskedCouplingBlock
                    _mask_type = 'checkerboard'
                    div_in = 1
                    mult_out = 2

                nonloc_scope.swap = not nonloc_scope.swap
                _mask_type += '1' if nonloc_scope.swap else '0'

                nnet = []
                if not first_resblock and preact:
                    if batchnorm: nnet.append(layers.MovingBatchNorm2d(initial_size[0]))
                    nnet.append(ACT_FNS[activation_fn](False))
                nnet.append(_weight_layer(fc)(initial_size[0] // div_in, idim, ks[0], 1, ks[0] // 2))
                if batchnorm: nnet.append(layers.MovingBatchNorm2d(idim))
                nnet.append(ACT_FNS[activation_fn](True))
                for i, k in enumerate(ks[1:-1]):
                    nnet.append(_weight_layer(fc)(idim, idim, k, 1, k // 2))
                    if batchnorm: nnet.append(layers.MovingBatchNorm2d(idim))
                    nnet.append(ACT_FNS[activation_fn](True))
                if dropout: nnet.append(nn.Dropout2d(dropout, inplace=True))
                nnet.append(_weight_layer(fc)(idim, initial_size[0] * mult_out, ks[-1], 1, ks[-1] // 2))
                if batchnorm: nnet.append(layers.MovingBatchNorm2d(initial_size[0]))

                return _block(initial_size[0], nn.Sequential(*nnet), mask_type=_mask_type)

        if init_layer is not None: chain.append(init_layer)
        if first_resblock and actnorm: chain.append(_actnorm(initial_size, fc))
        if first_resblock and fc_actnorm: chain.append(_actnorm(initial_size, True))

        if squeeze:
            c, h, w = initial_size
            for i in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc, first_resblock=first_resblock and (i == 0)))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            chain.append(layers.SqueezeLayer(2))
        else:
            for _ in range(n_blocks):
                if quadratic: chain.append(_quadratic_layer(initial_size, fc))
                chain.append(_resblock(initial_size, fc))
                if actnorm: chain.append(_actnorm(initial_size, fc))
                if fc_actnorm: chain.append(_actnorm(initial_size, True))
            # Use four fully connected layers at the end.
            if fc_end:
                for _ in range(fc_nblocks):
                    chain.append(_resblock(initial_size, True, fc_idim))
                    if actnorm or fc_actnorm: chain.append(_actnorm(initial_size, True))

        super(StackedCouplingBlocks, self).__init__(chain)
