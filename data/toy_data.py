import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader, TensorDataset
import sklearn.datasets
import math
import os
import time
import argparse
import pprint
from functools import partial
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------
# Data
# --------------------

def potential_fn(dataset):
    # NF paper table 1 energy functions
    w1 = lambda z: torch.sin(2 * math.pi * z[:,0] / 4)
    w2 = lambda z: 3 * torch.exp(-0.5 * ((z[:,0] - 1)/0.6)**2)
    w3 = lambda z: 3 * torch.sigmoid((z[:,0] - 1) / 0.3)

    if dataset == 'u1':
        return lambda z: 0.5 * ((torch.norm(z, p=2, dim=1) - 2) / 0.4)**2 - \
                                torch.log(torch.exp(-0.5*((z[:,0] - 2) / 0.6)**2) + \
                                          torch.exp(-0.5*((z[:,0] + 2) / 0.6)**2) + 1e-10)

    elif dataset == 'u2':
        return lambda z: 0.5 * ((z[:,1] - w1(z)) / 0.4)**2

    elif dataset == 'u3':
        return lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.35)**2) + \
                                     torch.exp(-0.5*((z[:,1] - w1(z) + w2(z))/0.35)**2) + 1e-10)

    elif dataset == 'u4':
        return lambda z: - torch.log(torch.exp(-0.5*((z[:,1] - w1(z))/0.4)**2) + \
                                     torch.exp(-0.5*((z[:,1] - w1(z) + w3(z))/0.35)**2) + 1e-10)

    else:
        raise RuntimeError('Invalid potential name to sample from.')

def sample_2d_data(dataset, n_samples):

    # z = torch.randn(n_samples, 2)

    if dataset == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(n_samples):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset
        # return torch.from_numpy(dataset).type(torch.float32)
        # return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        dataset = x + 0.1*z
        return dataset.cpu().numpy()

    elif dataset == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        # return torch.from_numpy(data)
        return data

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]
        # Add noise
        dataset = x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))
        return dataset.cpu().numpy()

    elif dataset == 'swissroll':
        data = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        data = torch.from_numpy(data)
        return data

    elif dataset == "circles":
        data = sklearn.datasets.make_circles(n_samples=n_samples, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        data = torch.from_numpy(data)
        return data

    elif dataset == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return torch.from_numpy(data)

    elif dataset == "line":
        x = rng.rand(n_samples) * 5 - 2.5
        y = x
        return np.stack((x, y), 1)

    elif dataset == "cos":
        x = rng.rand(n_samples) * 5 - 2.5
        y = np.sin(x) * 2.5
        return np.stack((x, y), 1)

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')
