import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn.gspaces import *
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.nn import EquivariantModule
from typing import List, Tuple, Any
import numpy as np

class Identity(nn.Module):

    def forward(self, x):
        return x


class FullSort(nn.Module):

    def forward(self, x):
        return torch.sort(x, 1)[0]


class MaxMin(nn.Module):

    def forward(self, x):
        b, d = x.shape
        max_vals = torch.max(x.view(b, d // 2, 2), 2)[0]
        min_vals = torch.min(x.view(b, d // 2, 2), 2)[0]
        return torch.cat([max_vals, min_vals], 1)


class LipschitzCube(nn.Module):

    def forward(self, x):
        return (x >= 1).to(x) * (x - 2 / 3) + (x <= -1).to(x) * (x + 2 / 3) + ((x > -1) * (x < 1)).to(x) * x**3 / 3


class SwishFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, beta):
        beta_sigm = torch.sigmoid(beta * x)
        output = x * beta_sigm
        ctx.save_for_backward(x, output, beta)
        return output / 1.1

    @staticmethod
    def backward(ctx, grad_output):
        x, output, beta = ctx.saved_tensors
        beta_sigm = output / x
        grad_x = grad_output * (beta * output + beta_sigm * (1 - beta * output))
        grad_beta = torch.sum(grad_output * (x * output - output * output)).expand_as(beta)
        return grad_x / 1.1, grad_beta / 1.1


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

class GeomSwish(EquivariantModule):

    def __init__(self, in_type: FieldType, inplace: bool = False):
        r"""

        Module that implements a pointwise Swish to every channel independently.
        The input representation is preserved by this operation and, therefore, it equals the output
        representation.

        Only representations supporting pointwise non-linearities are accepted as input field type.

        Args:
            in_type (FieldType):  the input field type
            inplace (bool, optional): can optionally do the operation in-place. Default: ``False``

        """

        assert isinstance(in_type.gspace, GeneralOnR2)

        super(GeomSwish, self).__init__()

        for r in in_type.representations:
            assert 'pointwise' in r.supported_nonlinearities, \
                'Error! Representation "{}" does not support "pointwise" non-linearity'.format(r.name)

        self.space = in_type.gspace
        self.in_type = in_type
        self.beta = nn.Parameter(torch.tensor([0.5]))

        # the representation in input is preserved
        self.out_type = in_type

        self._inplace = inplace

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        r"""

        Applies ReLU function on the input fields

        Args:
            input (GeometricTensor): the input feature map

        Returns:
            the resulting feature map after relu has been applied

        """

        assert input.type == self.in_type, "Error! the type of the input does not match the input type of this module"
        return GeometricTensor((input.tensor * torch.sigmoid_(input.tensor * F.softplus(self.beta))).div_(1.1), self.out_type)


    def evaluate_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size

        b, c, hi, wi = input_shape

        return b, self.out_type.size, hi, wi

    def check_equivariance(self, atol: float = 1e-6, rtol: float = 1e-5) -> List[Tuple[Any, float]]:

        c = self.in_type.size

        x = torch.randn(3, c, 10, 10)

        x = GeometricTensor(x, self.in_type)

        errors = []

        for el in self.space.testing_elements:
            out1 = self(x).transform_fibers(el)
            out2 = self(x.transform_fibers(el))

            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())

            assert torch.allclose(out1.tensor, out2.tensor, atol=atol, rtol=rtol), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

            errors.append((el, errs.mean()))

        return errors

    def extra_repr(self):
        return 'inplace={}, type={}'.format(
            self._inplace, self.in_type
        )

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.ReLU` module and set to "eval" mode.

        """

        self.eval()

        return torch.nn.ReLU(inplace=self._inplace)

if __name__ == '__main__':

    m = Swish()
    xx = torch.linspace(-5, 5, 1000).requires_grad_(True)
    yy = m(xx)
    dd, dbeta = torch.autograd.grad(yy.sum() * 2, [xx, m.beta])

    import matplotlib.pyplot as plt

    plt.plot(xx.detach().numpy(), yy.detach().numpy(), label='Func')
    plt.plot(xx.detach().numpy(), dd.detach().numpy(), label='Deriv')
    plt.plot(xx.detach().numpy(), torch.max(dd.detach().abs() - 1, torch.zeros_like(dd)).numpy(), label='|Deriv| > 1')
    plt.legend()
    plt.tight_layout()
    plt.show()
