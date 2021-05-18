import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn
from torch.nn import Parameter
import ipdb

__all__ = ['ActNorm1d', 'ActNorm2d', 'EquivariantActNorm1d', 'EquivariantActNorm2d']


class ActNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(ActNormNd, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self, x, logpx=None):
        c = x.size(1)
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().view(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))
                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.view(*self.shape).expand_as(x)
        weight = self.weight.view(*self.shape).expand_as(x)

        y = (x + bias) * torch.exp(weight)

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        assert self.initialized
        bias = self.bias.view(*self.shape).expand_as(y)
        weight = self.weight.view(*self.shape).expand_as(y)

        x = y * torch.exp(-weight) - bias

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        return self.weight.view(*self.shape).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))


class EquivariantActNormNd(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(EquivariantActNormNd, self).__init__()
        self.num_features = num_features
        print("Num features %d" %(num_features))
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))

    @property
    def shape(self):
        raise NotImplementedError

    def forward(self, x, logpx=None):
        in_type = x.type
        _, c, h, w = x.shape
        x = x.tensor
        c = x.size(1)
        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 1).contiguous().view(c, -1)
                batch_mean = torch.mean(x_t, dim=1)
                batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))
                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.view(*self.shape).expand_as(x)
        weight = self.weight.view(*self.shape).expand_as(x)
        # bias = self.bias.view(-1, c, h , w).expand_as(x)
        # weight = self.weight.view(-1, c, h , w).expand_as(x)

        y = (x + bias) * torch.exp(weight)
        y = enn.GeometricTensor(y.view(-1, c, h, w), in_type)

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        in_type = y.type
        _, c, h, w = y.shape
        y = y.tensor
        assert self.initialized
        bias = self.bias.view(*self.shape).expand_as(y)
        weight = self.weight.view(*self.shape).expand_as(y)
        # bias = self.bias.view(-1, c, h, w).expand_as(y)
        # weight = self.weight.view(-1, c, h, w).expand_as(y)

        x = y * torch.exp(-weight) - bias
        x = enn.GeometricTensor(x.view(-1, c, h, w), in_type)

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x)

    def _logdetgrad(self, x):
        _, c, h, w = x.shape
        # return self.weight.view(-1, c, h, w).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)
        return self.weight.view(*self.shape).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)

    def __repr__(self):
        return ('{name}({num_features})'.format(name=self.__class__.__name__, **self.__dict__))

class ActNorm1d(ActNormNd):

    @property
    def shape(self):
        return [1, -1]


class ActNorm2d(ActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]

class EquivariantActNorm1d(EquivariantActNormNd):

    @property
    def shape(self):
        return [1, -1]

class EquivariantActNorm2d(EquivariantActNormNd):

    @property
    def shape(self):
        return [1, -1, 1, 1]
