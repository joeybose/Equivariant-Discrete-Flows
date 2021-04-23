
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from .autoaugment import CIFAR10Policy

DATA_DIR = "./data/cifar-10-batches-py"

MEAN = np.array([125.3, 123.0, 113.9]) / 255.0  # = np.array([0.49137255, 0.48235294, 0.44666667])
STD = np.array([63.0, 62.1, 66.7]) / 255.0  # = np.array([0.24705882, 0.24352941, 0.26156863])

def AddNoise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    noise = x.new().resize_as_(x).uniform_()
    x = x * (nvals - 1) + noise
    x = x / nvals
    return x

class Cutout:

    """Randomly mask out a patch from an image.
    Args:
        size (int): The size of the square patch.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image
        Returns:
            Tensor: Image with a hole of dimension size x size cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.size // 2, 0, h)
        y2 = np.clip(y + self.size // 2, 0, h)
        x1 = np.clip(x - self.size // 2, 0, w)
        x2 = np.clip(x + self.size // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

########################################################################################################################


def build_cifar10_loaders(batch_size,
                          eval_batchsize,
                          validation=True,
                          num_workers=8,
                          augment=False,
                          reshuffle=True,
                          ):

    # train_normalize = transforms.Normalize(
    #     mean=[0.4914, 0.4822, 0.4465],
    #     std=[0.2023, 0.1994, 0.2010],
    # )

    # test_normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225],
    # )

    normalize = transforms.Normalize(
        mean=MEAN,
        std=STD,
    )

    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        AddNoise,
        normalize,
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            AddNoise,
            Cutout(16),
            # normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            AddNoise,
            # normalize,
        ])
        # train_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=True,
        download=True, transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR, train=False,
        download=True, transform=valid_transform,
    )

    if validation:

        valid_dataset = datasets.CIFAR10(
            root=DATA_DIR, train=True,
            download=True, transform=valid_transform,
        )
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(0.2 * num_train))

        if reshuffle:
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=eval_batchsize, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batchsize, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    n_inputs = 3
    n_classes = 10

    return train_loader, valid_loader, test_loader


