from typing import Any
from .toy_data import sample_2d_data, potential_fn
from .mnist_rot_code import data_loader_mnist_rot
from .mnist_fliprot_code import data_loader_mnist_fliprot
from .mnist12k_code import data_loader_mnist12k
import argparse


def create_dataset(arg_parse: argparse.Namespace, dataset_type: str, *args: Any, **kwargs: Any):
    dataset_type = dataset_type.lower()
    if dataset_type in ["8gaussians", "2spirals", "checkerboard", "rings", "pinwheel", "swissroll", "circles", "line", "cos"]:
        return sample_2d_data(dataset_type, arg_parse.batch_size)
    elif dataset_type in ["u1", "u2", "u3", "u4"]:
        return potential_fn(dataset_type)
    elif dataset_type == "mnist_rot":
        arg_parse.im_dim = 1
        arg_parse.n_classes = 10
        if arg_parse.validation:
            train_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("train",
                                                                              arg_parse,
                                                                              arg_parse.batch_size,
                                                                              rot_interpol_augmentation=arg_parse.augment,
                                                                              interpolation=arg_parse.interpolation,
                                                                              reshuffle_seed=arg_parse.seed)
            valid_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("valid",
                                                                              arg_parse,
                                                                              arg_parse.val_batchsize,
                                                                              rot_interpol_augmentation=False,
                                                                              interpolation=arg_parse.interpolation,
                                                                              reshuffle_seed=arg_parse.seed)
        else:
            train_loader, _, _ = data_loader_mnist_rot.build_mnist_rot_loader("trainval",
                                                                              arg_parse,
                                                                              arg_parse.batch_size,
                                                                              rot_interpol_augmentation=arg_parse.augment,
                                                                              interpolation=arg_parse.interpolation,
                                                                              reshuffle_seed=None)
            valid_loader = False

        test_loader, n_inputs, n_outputs = data_loader_mnist_rot.build_mnist_rot_loader("test",
                                                                                        arg_parse,
                                                                                        arg_parse.val_batchsize,
                                                                                        rot_interpol_augmentation=False)
        return train_loader, test_loader
    elif dataset_type == "mnist_fliprot":
        arg_parse.im_dim = 1
        arg_parse.n_classes = 10
        if arg_parse.validation:
            train_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("train",
                                                                                  arg_parse,
                                                                                  arg_parse.batch_size,
                                                                                  rot_interpol_augmentation=arg_parse.augment,
                                                                                  interpolation=arg_parse.interpolation,
                                                                                  reshuffle_seed=arg_parse.seed)
            valid_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("valid",
                                                                                  arg_parse,
                                                                                  arg_parse.val_batchsize,
                                                                                  rot_interpol_augmentation=False,
                                                                                  interpolation=arg_parse.interpolation,
                                                                                  reshuffle_seed=arg_parse.seed)
        else:
            train_loader, _, _ = data_loader_mnist_fliprot.build_mnist_rot_loader("trainval",
                                                                                  arg_parse,
                                                                                  arg_parse.batch_size,
                                                                                  rot_interpol_augmentation=arg_parse.augment,
                                                                                  interpolation=arg_parse.interpolation,
                                                                                  reshuffle_seed=None)
            valid_loader = False

        test_loader, n_inputs, n_outputs = data_loader_mnist_fliprot.build_mnist_rot_loader("test",
                                                                                            arg_parse,
                                                                                            arg_parse.val_batchsize,
                                                                                            rot_interpol_augmentation=False)
        return train_loader, test_loader
    elif dataset_type == "mnist12k":
        arg_parse.im_dim = 1
        arg_parse.n_classes = 10
        if arg_parse.validation:
            train_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("train",
                                                                            arg_parse,
                                                                            arg_parse.batch_size,
                                                                            rot_interpol_augmentation=arg_parse.augment,
                                                                            interpolation=arg_parse.interpolation,
                                                                            reshuffle_seed=arg_parse.seed)
            valid_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("valid",
                                                                            arg_parse,
                                                                            arg_parse.val_batchsize,
                                                                            rot_interpol_augmentation=False,
                                                                            interpolation=arg_parse.interpolation,
                                                                            reshuffle_seed=arg_parse.seed)
        else:
            train_loader, _, _ = data_loader_mnist12k.build_mnist12k_loader("trainval",
                                                                            arg_parse,
                                                                            arg_parse.batch_size,
                                                                            rot_interpol_augmentation=arg_parse.augment,
                                                                            interpolation=arg_parse.interpolation,
                                                                            reshuffle_seed=None)
            valid_loader = False

        test_loader, n_inputs, n_outputs = data_loader_mnist12k.build_mnist12k_loader("test",
                                                                                      arg_parse,
                                                                                      arg_parse.val_batchsize,
                                                                                      # rot_interpol_augmentation=False
                                                                                      # interpolation=interpolation,
                                                                                      )
        return train_loader, test_loader
    else:
        raise ValueError(f"Unknown dataset type: '{dataset_type}'.")
