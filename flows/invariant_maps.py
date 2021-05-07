import torch
from e2cnn import gspaces
from e2cnn import nn as enn
import ipdb
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import Resize
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose
import numpy as np
from PIL import Image
from flows.flow_utils import *

FIBERS = {
    "trivial": trivial_fiber,
    "quotient": quotient_fiber,
    "regular": regular_fiber,
    "irrep": irrep_fiber,
    "mixed1": mixed1_fiber,
    "mixed2": mixed2_fiber,
}

class InvariantCNNBlock(torch.nn.Module):

    def __init__(self, input_size, in_type, field_type, out_fiber,
                 activation_fn, hidden_size, group_action_type):

        super(InvariantCNNBlock, self).__init__()
        _, self.c, self.h, self.w = input_size
        ngf = 16
        self.group_action_type = group_action_type
        feat_type_in = enn.FieldType(self.group_action_type, self.c*[self.group_action_type.trivial_repr])
        feat_type_hid = FIBERS[out_fiber](group_action_type, hidden_size,
                                          field_type, fixparams=True)
        feat_type_out = enn.FieldType(self.group_action_type, 128*[self.group_action_type.regular_repr])


        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = feat_type_in

        self.block1 = enn.SequentialModule(
            enn.R2Conv(feat_type_in, feat_type_hid, kernel_size=5, padding=0),
            enn.InnerBatchNorm(feat_type_hid),
            activation_fn(feat_type_hid, inplace=True),
        )

        self.pool1 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(feat_type_hid, sigma=0.66, stride=2)
        )


        self.block2 = enn.SequentialModule(
            enn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=5),
            enn.InnerBatchNorm(feat_type_hid),
            activation_fn(feat_type_hid, inplace=True),
        )

        self.pool2 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(feat_type_hid, sigma=0.66, stride=2)
        )

        self.block3 = enn.SequentialModule(
            enn.R2Conv(feat_type_hid, feat_type_out, kernel_size=3, padding=1),
            enn.InnerBatchNorm(feat_type_out),
            activation_fn(feat_type_out, inplace=True),
        )

        self.pool3 = enn.PointwiseAvgPoolAntialiased(feat_type_out, sigma=0.66, stride=1, padding=0)

        self.gpool = enn.GroupPooling(feat_type_out)

        self.gc = self.gpool.out_type.size
        self.gen = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.gc, ngf, kernel_size=4, stride=1,padding=0),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(ngf, int(ngf/2), kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(int(ngf/2)),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(int(ngf/2), self.c, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, input):
        # apply each equivariant block

        # Each layer has an input and an output type
        # As a result, consecutive layers need to have matching input/output types
        # ipdb.set_trace()
        # x = enn.GeometricTensor(input, self.input_type)
        x = self.block1(input)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # Upsample DCGAN style
        x = self.gen(x)
        x = enn.GeometricTensor(x, self.input_type)

        return x.tensor

class InvariantCNNBlock2(torch.nn.Module):

    def __init__(self, input_size, in_type, field_type, out_fiber,
                 activation_fn, hidden_size, group_action_type):

        super(InvariantCNNBlock2, self).__init__()
        _, self.c, self.h, self.w = input_size
        self.group_action_type = group_action_type
        feat_type_in = enn.FieldType(self.group_action_type, self.c*[self.group_action_type.trivial_repr])
        feat_type_hid = FIBERS[out_fiber](group_action_type, hidden_size,
                                          field_type, fixparams=True)
        feat_type_out = enn.FieldType(self.group_action_type, 128*[self.group_action_type.regular_repr])


        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = feat_type_in

        self.block1 = enn.SequentialModule(
            enn.R2Conv(feat_type_in, feat_type_hid, kernel_size=5, padding=0),
            enn.InnerBatchNorm(feat_type_hid),
            activation_fn(feat_type_hid, inplace=True),
        )

        self.pool1 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(feat_type_hid, sigma=0.66, stride=2)
        )


        self.block2 = enn.SequentialModule(
            enn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=5),
            enn.InnerBatchNorm(feat_type_hid),
            activation_fn(feat_type_hid, inplace=True),
        )

        self.pool2 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(feat_type_hid, sigma=0.66, stride=2)
        )

        self.block3 = enn.SequentialModule(
            enn.R2Conv(feat_type_hid, feat_type_out, kernel_size=3, padding=1),
            enn.InnerBatchNorm(feat_type_out),
            activation_fn(feat_type_out, inplace=True),
        )

        self.pool3 = enn.PointwiseAvgPoolAntialiased(feat_type_out, sigma=0.66, stride=1, padding=0)

        self.gpool = enn.GroupPooling(feat_type_out)

        self.gc = self.gpool.out_type.size
        self.init_size = self.h // 4
        self.l1 = torch.nn.Sequential(torch.nn.Linear(self.gc, 128 * self.init_size ** 2))

        self.gen = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, self.c, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, input):
        # apply each equivariant block

        # Each layer has an input and an output type
        # As a result, consecutive layers need to have matching input/output types
        # x = enn.GeometricTensor(input, self.input_type)
        x = self.block1(input)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # Upsample DCGAN style
        out = self.l1(x.squeeze())
        x = out.view(out.shape[0], 128, self.init_size, self.init_size)
        x = self.gen(x)
        x = enn.GeometricTensor(x, self.input_type)

        return x

class MnistInvariantCNNBlock(torch.nn.Module):

    def __init__(self, input_size, in_type, field_type, out_fiber,
                 activation_fn, hidden_size, group_action_type):

        super(MnistInvariantCNNBlock, self).__init__()
        _, self.c, self.h, self.w = input_size
        self.group_action_type = group_action_type
        feat_type_in = enn.FieldType(self.group_action_type, self.c*[self.group_action_type.trivial_repr])
        feat_type_hid = FIBERS[out_fiber](group_action_type, hidden_size,
                                          field_type, fixparams=True)
        feat_type_out = enn.FieldType(self.group_action_type, 128*[self.group_action_type.regular_repr])


        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = feat_type_in

        self.block1 = enn.SequentialModule(
            enn.R2Conv(feat_type_in, feat_type_hid, kernel_size=5, padding=0),
            enn.InnerBatchNorm(feat_type_hid),
            activation_fn(feat_type_hid, inplace=True),
        )

        self.pool1 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(feat_type_hid, sigma=0.66, stride=2)
        )


        self.block2 = enn.SequentialModule(
            enn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=5, padding=1),
            enn.InnerBatchNorm(feat_type_hid),
            activation_fn(feat_type_hid, inplace=True),
        )

        self.pool2 = enn.SequentialModule(
            enn.PointwiseAvgPoolAntialiased(feat_type_hid, sigma=0.66, stride=2)
        )

        self.block3 = enn.SequentialModule(
            enn.R2Conv(feat_type_hid, feat_type_out, kernel_size=3, padding=1),
            enn.InnerBatchNorm(feat_type_out),
            activation_fn(feat_type_out, inplace=True),
        )

        self.pool3 = enn.PointwiseAvgPoolAntialiased(feat_type_out, sigma=0.66, stride=1, padding=0)

        self.gpool = enn.GroupPooling(feat_type_out)

        self.gc = self.gpool.out_type.size
        self.init_size = self.h // 4
        self.l1 = torch.nn.Sequential(torch.nn.Linear(self.gc, 128 * self.init_size ** 2))

        self.gen = torch.nn.Sequential(
            torch.nn.BatchNorm2d(128),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 128, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, self.c, 3, stride=1, padding=1),
            torch.nn.Tanh(),
        )

    def forward(self, input):
        # apply each equivariant block

        # Each layer has an input and an output type
        # As a result, consecutive layers need to have matching input/output types
        # x = enn.GeometricTensor(input, self.input_type)
        x = self.block1(input)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        # pool over the spatial dimensions
        x = self.pool3(x)

        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        # Upsample DCGAN style
        out = self.l1(x.squeeze())
        x = out.view(out.shape[0], 128, self.init_size, self.init_size)
        x = self.gen(x)
        x = enn.GeometricTensor(x, self.input_type)

        return x
