import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
import torch.nn.init as init
import math
import os
import math
import argparse
import pprint
import numpy as np
import copy
from e2cnn import gspaces
from e2cnn import nn as enn

CHANNELS_CONSTANT = 1

def regular_fiber(gspace: gspaces.GeneralOnR2, planes: int, field_type: int =
                  0, fixparams: bool = True):
    """ build a regular fiber with the specified number of channels"""
    assert gspace.fibergroup.order() > 0
    N = gspace.fibergroup.order()
    planes = planes / N
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    planes = int(planes)

    if planes % 2 != 0:
        planes += 1

    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def irrep_fiber(gspace: gspaces.GeneralOnR2, planes: int, field_type: int = 0,
                fixparams: bool = True):
    """ build a irrep fiber with the specified number of channels"""
    assert gspace.fibergroup.order() < 0
    N = gspace.fibergroup.order()
    planes = int(planes)

    if planes % 2 != 0:
        planes += 1
    return enn.FieldType(gspace, [gspace.irrep(0)] * planes)

def quotient_fiber(gspace: gspaces.GeneralOnR2, planes: int, field_type: int =
                   0, fixparams: bool = True):
    """ build a quotient fiber with the specified number of channels"""
    N = gspace.fibergroup.order()
    assert N > 0
    if isinstance(gspace, gspaces.FlipRot2dOnR2):
        n = N/2
        subgroups = []
        for axis in [0, round(n/4), round(n/2)]:
            subgroups.append((int(axis), 1))
    elif isinstance(gspace, gspaces.Rot2dOnR2):
        assert N % 4 == 0
        # subgroups = [int(round(N/2)), int(round(N/4))]
        subgroups = [2, 4]
    elif isinstance(gspace, gspaces.Flip2dOnR2):
        subgroups = [2]
    else:
        raise ValueError(f"Space {gspace} not supported")

    rs = [gspace.quotient_repr(subgroup) for subgroup in subgroups]
    size = sum([r.size for r in rs])
    planes = planes / size
    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)
    planes = int(planes)
    return enn.FieldType(gspace, rs * planes).sorted()


def trivial_fiber(gspace: gspaces.GeneralOnR2, planes: int, field_type: int =
                  0, fixparams: bool = True):
    """ build a trivial fiber with the specified number of channels"""

    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order() * CHANNELS_CONSTANT)
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


def mixed_fiber(gspace: gspaces.GeneralOnR2, planes: int, ratio: float,
                field_type: int = 0, fixparams: bool = True):

    N = gspace.fibergroup.order()
    assert N > 0
    if isinstance(gspace, gspaces.FlipRot2dOnR2):
        subgroup = (0, 1)
    elif isinstance(gspace, gspaces.Flip2dOnR2):
        subgroup = 1
    else:
        raise ValueError(f"Space {gspace} not supported")

    qr = gspace.quotient_repr(subgroup)
    rr = gspace.regular_repr

    planes = planes / rr.size

    if fixparams:
        planes *= math.sqrt(N * CHANNELS_CONSTANT)

    r_planes = int(planes * ratio)
    q_planes = int(2*planes * (1-ratio))

    return enn.FieldType(gspace, [rr] * r_planes + [qr] * q_planes).sorted()


def mixed1_fiber(gspace: gspaces.GeneralOnR2, planes: int, field_type: int = 0,
                 fixparams: bool = True):
    return mixed_fiber(gspace=gspace, planes=planes, ratio=0.5,
                       field_type=field_type, fixparams=fixparams)


def mixed2_fiber(gspace: gspaces.GeneralOnR2, planes: int, field_type: int = 0,
                 fixparams: bool = True):
    return mixed_fiber(gspace=gspace, planes=planes, ratio=0.25,
                       field_type=field_type, fixparams=fixparams)
