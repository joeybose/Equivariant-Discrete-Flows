
import math
import torch
from torch.distributions.kl import register_kl
_EPS = 1e-7

# class HypersphericalUniform(torch.distributions.Distribution):

    # arg_constraints = {
        # "dim": torch.distributions.constraints.positive_integer,
    # }

    # def __init__(self, dim, device="cpu", dtype=torch.float32, validate_args=None):
        # self.dim = dim if isinstance(dim, torch.Tensor) else torch.tensor(dim, device=device)
        # super().__init__(validate_args=validate_args)
        # self.device, self.dtype = device, dtype

    # def rsample(self, sample_shape=()):
        # v = torch.empty(
            # sample_shape + (self.dim,), device=self.device, dtype=self.dtype
        # ).normal_()
        # return v / (v.norm(dim=-1, keepdim=True) + _EPS)

    # def log_prob(self, value):
        # return torch.full_like(
            # value[..., 0],
            # math.lgamma(self.dim / 2)
            # - (math.log(2) + (self.dim / 2) * math.log(math.pi)),
            # device=self.device,
            # dtype=self.dtype,
        # )

    # def entropy(self):
        # return -self.log_prob(torch.empty(1))

    # def __repr__(self):
        # return "HypersphericalUniform(dim={}, device={}, dtype={})".format(
            # self.dim, self.device, self.dtype
        # )

class HypersphericalUniform(torch.distributions.Distribution):

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val if isinstance(val, torch.device) else torch.device(val)

    def __init__(self, dim, validate_args=None, device="cpu"):
        super(HypersphericalUniform, self).__init__(
            torch.Size([dim]), validate_args=validate_args
        )
        self._dim = dim
        self.device = device

    def sample(self, shape=torch.Size()):
        output = (
            torch.distributions.Normal(0, 1)
            .sample(
                (shape if isinstance(shape, torch.Size) else torch.Size([shape]))
                + torch.Size([self._dim + 1])
            )
            .to(self.device)
        )

        return output / output.norm(dim=-1, keepdim=True)

    def entropy(self):
        return self.__log_surface_area()

    def log_prob(self, x):
        return -torch.ones(x.shape[:-1], device=self.device) * self.__log_surface_area()

    def __log_surface_area(self):
        if torch.__version__ >= "1.0.0":
            lgamma = torch.lgamma(torch.tensor([(self._dim + 1) / 2]).to(self.device))
        else:
            lgamma = torch.lgamma(
                torch.Tensor([(self._dim + 1) / 2], device=self.device)
            )
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - lgamma

