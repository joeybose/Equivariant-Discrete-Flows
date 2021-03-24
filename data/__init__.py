from typing import Any
from .toy_data import sample_2d_data, potential_fn
import argparse


def create_dataset(arg_parse: argparse.Namespace, dataset_type: str, *args: Any, **kwargs: Any):
    dataset_type = dataset_type.lower()
    if dataset_type in ["8gaussians", "2spirals", "checkerboard", "rings", "pinwheel", "swissroll", "circles", "line", "cos"]:
        arg_parse.input_dim = 2
        return sample_2d_data(dataset_type, arg_parse.nsamples)
    elif dataset_type in ["u1", "u2", "u3", "u4"]:
        arg_parse.input_dim = 2
        return potential_fn(dataset_type)
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
