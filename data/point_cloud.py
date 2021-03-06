import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class SpatialMNIST(Dataset):
    def __init__(self, data_dir, train):
        super(SpatialMNIST, self).__init__()

        self.data = datasets.MNIST(data_dir, train=train, download=True, transform=transforms.ToTensor())
        self.grid = np.stack(np.meshgrid(range(28), range(27,-1,-1)), axis=-1).reshape([-1,2])

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        img = img.numpy().reshape([-1])
        p = img / img.sum()
        replace = True if (sum(p > 0) < 50) else False
        ind = np.random.choice(784, 50, p=p, replace=replace)
        x = self.grid[ind].copy().astype(np.float32)
        x += np.random.uniform(0., 1., (50, 2))
        # normalize
        x /= 28. # [0, 1]

        return x

    def __len__(self):
        return len(self.data)

class ModelNet(Dataset):
    NUM_POINTS = 10000
    label_dict = {'airplane': 0,
              'bathtub': 1,
              'bed': 2,
              'bench': 3,
              'bookshelf': 4,
              'bottle': 5,
              'bowl': 6,
              'car': 7,
              'chair': 8,
              'cone': 9,
              'cup': 10,
              'curtain': 11,
              'desk': 12,
              'door': 13,
              'dresser': 14,
              'flower_pot': 15,
              'glass_box': 16,
              'guitar': 17,
              'keyboard': 18,
              'lamp': 19,
              'laptop': 20,
              'mantel': 21,
              'monitor': 22,
              'night_stand': 23,
              'person': 24,
              'piano': 25,
              'plant': 26,
              'radio': 27,
              'range_hood': 28,
              'sink': 29,
              'sofa': 30,
              'stairs': 31,
              'stool': 32,
              'table': 33,
              'tent': 34,
              'toilet': 35,
              'tv_stand': 36,
              'vase': 37,
              'wardrobe': 38,
              'xbox': 39,
              }
    def __init__(self, data_dir, category, set_size, train):
        super(ModelNet, self).__init__()

        with h5py.File(data_dir, 'r') as f:
            train_cloud = np.array(f['tr_cloud'])
            train_labels = np.array(f['tr_labels'])
            test_cloud = np.array(f['test_cloud'])
            test_labels = np.array(f['test_labels'])

        if train:
            data = train_cloud
            label = train_labels
        else:
            data = test_cloud
            label = test_labels

        if category != 'all':
            assert category in ModelNet.label_dict
            inds = np.where(label == ModelNet.label_dict[category])[0]
            data = data[inds]
            label = label[inds]
        self.data = data
        self.set_size = set_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # subsample
        sel = np.random.choice(x.shape[0], self.set_size, replace=False)
        x = x[sel]
        # preprocess
        x += np.random.randn(*x.shape) * 0.001
        x_max = np.max(x)
        x_min = np.min(x)
        x = (x - x_min) / (x_max - x_min)
        x -= np.mean(x, axis=0)

        return x

# def get_loader(args):
    # if args.data == 'spatial_mnist':
        # trainset = SpatialMNIST(args.data_dir, True)
        # testset = SpatialMNIST(args.data_dir, False)
    # elif args.data == 'modelnet':
        # trainset = ModelNet(args.data_dir, args.category, args.set_size, True)
        # testset = ModelNet(args.data_dir, args.category, args.set_size, False)
    # else:
        # raise Exception()
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # return trainloader, testloader
