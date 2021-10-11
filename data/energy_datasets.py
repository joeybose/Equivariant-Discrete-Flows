import numpy as np
import torch

class BatchIterator(object):
    def __init__(self, n_elems, n_batch):
        self._indices = np.arange(n_elems)
        self._n_elems = n_elems
        self._n_batch = n_batch
        self._pos = 0
        self._reset()

    def _reset(self):
        self._pos = 0
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        # subtract n_batch to have a constant batch size
        if self._pos >= self._n_elems - self._n_batch:
            self._reset()
            raise StopIteration
        n_collected = min(self._n_batch, self._n_elems - self._pos)
        batch = self._indices[self._pos : self._pos + n_collected]
        self._pos = self._pos + n_collected
        return batch

    def next(self):
        return self.__next__()

class IndexBatchIterator(object):
    def __init__(self, n_elems, n_batch):
        """
            Produces batches of length `n_batch` of an index set
            `[1, ..., n_elems]` which are sampled randomly without
            replacement.

            If `n_elems` is not a multiple of `n_batch` the last sampled
            batch will be truncated.

            After the iteration throw `StopIteration` its random seed
            will be reset.

            Parameters:
            -----------
            n_elems : Integer
                Number of elements in the index set.
            n_batch : Integer
                Number of batch elements sampled.

        """
        self._indices = np.arange(n_elems)
        self._n_elems = n_elems
        self._n_batch = n_batch
        self._pos = 0
        self._reset()

    def _reset(self):
        self._pos = 0
        np.random.shuffle(self._indices)

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= self._n_elems:
            self._reset()
            raise StopIteration
        n_collected = min(self._n_batch, self._n_elems - self._pos)
        batch = self._indices[self._pos : self._pos + n_collected]
        self._pos = self._pos + n_collected
        return batch

    def __len__(self):
        return self._n_elems // self._n_batch

    def next(self):
        return self.__next__()

def compute_mean(samples, n_particles, n_dimensions, keepdim=False):
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        mean = torch.mean(samples, dim=1, keepdim=keepdim)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        mean = samples.mean(axis=1, keepdims=keepdim)
    return mean


def remove_mean(samples, n_particles, n_dimensions):
    shape = samples.shape
    if isinstance(samples, torch.Tensor):
        samples = samples.view(-1, n_particles, n_dimensions)
        samples = samples - torch.mean(samples, dim=1, keepdim=True)
        samples = samples.view(*shape)
    else:
        samples = samples.reshape(-1, n_particles, n_dimensions)
        samples = samples - samples.mean(axis=1, keepdims=True)
        samples = samples.reshape(*shape)
    return samples

class Sampler(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def _sample_with_temperature(self, n_samples, temperature):
        raise NotImplementedError()

    def _sample(self, n_samples):
        raise NotImplementedError()

    def sample(self, n_samples, temperature=None):
        if temperature is not None:
            return self._sample_with_temperature(n_samples, temperature)
        else:
            return self._sample(n_samples)

class Energy(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x)[0]


class MeanFreeNormalPrior(Energy, Sampler):
    def __init__(self, dim, n_particles):
        super().__init__(dim)
        self._dim = dim
        self._n_particles = n_particles
        self._spacial_dims = dim // n_particles

    def _energy(self, x):
        x = self._remove_mean(x)
        return 0.5 * x.pow(2).sum(dim=-1, keepdim=True)

    def sample(self, n_samples, temperature=1.):
        x = torch.Tensor(n_samples, self._n_particles, self._spacial_dims).normal_()
        return self._remove_mean(x)

    def _remove_mean(self, x):
        x = x.view(-1, self._n_particles, self._spacial_dims)
        x = x - torch.mean(x, dim=1, keepdim=True)
        return x.view(-1, self.dim).cuda()
