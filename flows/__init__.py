from typing import Any
from .flows import *
from .equivariant_flows import *
from .toy_resflow import *
import argparse
import ipdb

kwargs_flows = {'MAFRealNVP': MAFRealNVP, 'RealNVP': RealNVP, "Toy": toy_flow, "Simple":package_realnvp, "toy_resflow": ToyResFlow, "E_RealNVP": EquivariantRealNVP}

def create_flow(arg_parse: argparse.Namespace, model_type: str, *args: Any, **kwargs: Any):
    flow_model = kwargs_flows[model_type](arg_parse, arg_parse.n_blocks,
                                          arg_parse.input_dim,
                                          arg_parse.hidden_dim,
                                          arg_parse.num_layers).to(arg_parse.dev)
    return flow_model
