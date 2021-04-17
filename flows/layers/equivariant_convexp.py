from torch.nn.functional import conv2d, pad

from e2cnn.nn import init
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor, R2Conv
from e2cnn.gspaces import *
from typing import Callable, Union, Tuple, List
import ipdb
from flows.flow_helpers import *
import torch
from torch.nn import Parameter
import numpy as np
import math


__all__ = ["R2EquivariantConv"]

class R2EquivariantConv(R2Conv):

    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int,
                 padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 padding_mode: str = 'zeros',
                 groups: int = 1,
                 bias: bool = True,
                 basisexpansion: str = 'blocks',
                 sigma: Union[List[float], float] = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 ):

        assert in_type.gspace == out_type.gspace
        assert isinstance(in_type.gspace, GeneralOnR2)

        super().__init__(in_type=in_type,
                         out_type=out_type,
                         kernel_size=kernel_size,
                         padding=padding,
                         stride=stride,
                         dilation=dilation,
                         padding_mode=padding_mode,
                         groups=groups,
                         bias=bias,
                         basisexpansion=basisexpansion,
                         sigma=sigma,
                         frequencies_cutoff=frequencies_cutoff,
                         rings=rings,
                         maximum_offset=maximum_offset,
                         recompute=recompute,
                         basis_filter=basis_filter,
                         initialize=initialize)
        self.terms = 10
        self.dynamic_truncation = 0

    def forward(self, input: GeometricTensor, inverse=False):
            r"""
            Convolve the input with the expanded filter and bias.

            Args:
                input (GeometricTensor): input feature field transforming according to ``in_type``
            Returns:
                output feature field transforming according to ``out_type``

            """
            assert input.type == self.in_type

            if not self.training:
                _filter = self.filter
                _bias = self.expanded_bias
            else:
                # retrieve the filter and the bias
                _filter, _bias = self.expand_parameters()

            if inverse:
                ipdb.set_trace()
                _filter = -_filter #Inverse is just negative of the Matrix Exp
            # use it for convolution and return the result
            output = input.tensor
            product = input.tensor

            if self.padding_mode == 'zeros':
                for i in range(1, self.terms + 1):
                    product = conv2d(product, _filter, padding=self.padding,
                                     stride=self.stride, dilation=self.dilation,
                                     groups=self.groups, bias=_bias) / i
                    output = output + product

                    if self.dynamic_truncation != 0 and i > 5:
                        if product.abs().max().item() < dynamic_truncation:
                            break
            else:
                for i in range(1, self.terms + 1):
                    product = conv2d(pad(product,
                                         self._reversed_padding_repeated_twice,
                                         self.padding_mode), _filter,
                                     stride=self.stride,
                                     dilation=self.dilation, padding=(0,0),
                                     groups=self.groups, bias=_bias) / i

                    output = output + product

                    if self.dynamic_truncation != 0 and i > 5:
                        if product.abs().max().item() < dynamic_truncation:
                            break

            return GeometricTensor(output, self.out_type)

