from torch._six import container_abcs
from itertools import repeat
import torch
from e2cnn import gspaces
from e2cnn import nn as enn
import ipdb

def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

def check_equivariance(r2_act, out_type, data, func, data_type=None):
    input_type = enn.FieldType(r2_act, [r2_act.trivial_repr])
    if data_type == 'GeomTensor':
        data = enn.GeometricTensor(data.view(-1, 1, 1, 2), input_type)
    for g in r2_act.testing_elements:
        output = func(data)
        if data_type == 'GeomTensor':
            rg_output = enn.GeometricTensor(output.tensor.view(-1,1,1,2).cpu(),
                                         out_type).transform(g)
            data = enn.GeometricTensor(data.tensor.view(-1, 1, 1, 2).cpu(), input_type)
            x_transformed = enn.GeometricTensor(data.transform(g).tensor.view(-1,1,1,2),
                                                input_type)
        else:
            rg_output = enn.GeometricTensor(output.view(-1,1,1,2).cpu(),
                                            out_type).transform(g)
            data = enn.GeometricTensor(data.view(-1, 1, 1, 2).cpu(), input_type)
            x_transformed = data.transform(g).tensor.view(-1,1,1,2)

        output_rg = func(x_transformed)
        # Equivariance Condition
        if data_type == 'GeomTensor':
            output_rg = enn.GeometricTensor(output_rg.tensor.cpu(), out_type)
            data = enn.GeometricTensor(data.tensor.squeeze().view(-1,1,1,2),input_type)
        else:
            output_rg = enn.GeometricTensor(output_rg.view(-1,1,1,2).cpu(), out_type)
            data = data.tensor.squeeze()
        assert torch.allclose(rg_output.tensor.cpu().squeeze(), output_rg.tensor.squeeze(), atol=1e-5), g

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)
