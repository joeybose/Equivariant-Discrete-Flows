from typing import Any
from .flows import *
from .equivariant_flows import *
from .toy_resflow import *
from .resflow import *
from .equivariant_resflow import *
import argparse
import ipdb

kwargs_flows = {'MAFRealNVP': MAFRealNVP, 'RealNVP': RealNVP, "Toy": toy_flow,
                "Simple":package_realnvp, "toy_resflow": ToyResFlow, "resflow":
                ResidualFlow, "FiberRealNVP": FiberRealNVP, "E_toy_resflow":
                EquivariantToyResFlow, "resflow": ResidualFlow, "E_resflow":
                EquivariantResidualFlow, "E_convexp": EquivariantConvExp}

def create_flow(arg_parse: argparse.Namespace, model_type: str, *args: Any, **kwargs: Any):
    # if arg_parse.dataset in ["8gaussians", "2spirals", "checkerboard", "rings", "pinwheel", "swissroll", "circles", "line", "cos"]:
    if 'resflow' not in model_type:
        flow_model = kwargs_flows[model_type](arg_parse, arg_parse.n_blocks,
                                              arg_parse.input_size,
                                              arg_parse.hidden_dim,
                                              arg_parse.num_layers).to(arg_parse.dev)
    else:
        flow_model = kwargs_flows[model_type](
            arg_parse,
            arg_parse.input_size,
            n_blocks=list(map(int, arg_parse.n_blocks.split('-'))),
            intermediate_dim=arg_parse.idim,
            factor_out=arg_parse.factor_out,
            quadratic=arg_parse.quadratic,
            init_layer=arg_parse.init_layer,
            actnorm=arg_parse.actnorm,
            fc_actnorm=arg_parse.fc_actnorm,
            batchnorm=arg_parse.batchnorm,
            dropout=arg_parse.dropout,
            fc=arg_parse.fc,
            coeff=arg_parse.coeff,
            vnorms=arg_parse.vnorms,
            n_lipschitz_iters=arg_parse.n_lipschitz_iters,
            sn_atol=arg_parse.sn_tol,
            sn_rtol=arg_parse.sn_tol,
            n_power_series=arg_parse.n_power_series,
            n_dist=arg_parse.n_dist,
            n_samples=arg_parse.n_samples,
            kernels=arg_parse.kernels,
            activation_fn=arg_parse.act,
            fc_end=arg_parse.fc_end,
            fc_idim=arg_parse.fc_idim,
            n_exact_terms=arg_parse.n_exact_terms,
            preact=arg_parse.preact,
            neumann_grad=arg_parse.neumann_grad,
            grad_in_forward=arg_parse.mem_eff,
            first_resblock=arg_parse.first_resblock,
            learn_p=arg_parse.learn_p,
            classification=arg_parse.task in ['classification', 'hybrid'],
            classification_hdim=arg_parse.cdim,
            n_classes=arg_parse.n_classes,
            block_type=arg_parse.block,
        ).to(arg_parse.dev)
    return flow_model
