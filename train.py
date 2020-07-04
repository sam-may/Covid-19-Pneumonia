import setGPU

import os, sys
import argparse
import numpy
import glob
import h5py
import json
import utils
import models
import train_helper

# CLI
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_extra_slices", 
    help="extra slices above and below input", 
    type=int, 
    default=0
)
parser.add_argument(
    "--tag", 
    help="tag to identify this set", 
    type=str, 
    default=""
)
parser.add_argument(
    "--input", 
    help="input hdf5", 
    type=str
)
parser.add_argument(
    "--input_metadata", 
    help="json file with metadata", 
    type=str
)
parser.add_argument(
    "--max_epochs", 
    help="maximum number of training epochs", 
    type=int, 
    default=20
)
parser.add_argument(
    "--random_seed",
    help="random seed for test/train split",
    type=int,
    default=0
)
parser.add_argument(
    "--load_weights",
    help="summary json of already trained model",
    type=str
)
args = parser.parse_args()

# Initialize training functions
helper = train_helper.TrainHelper(
    n_extra_slices=args.n_extra_slices,
    input=args.input,
    input_metadata=args.input_metadata,
    tag=args.tag,
    train_frac=0.7,
    fast=False,
    random_seed=args.random_seed,
    max_epochs=args.max_epochs
)

# If a summary json is supplied, just load weights (don't train)
if args.load_weights is not None:
    print("[train.py] Loading model from file: %s"
            % args.load_weights)
    with open(args.load_weights, "r") as f_in:
        summary = json.load(f_in)

    unet_config = summary["config"]
    model = models.unet(unet_config)

    # Load weights
    helper.load_weights(model, unet_config, summary["weights"])

# Otherwise, train a model from scratch
else:
    # Initialize model
    unet_config = {
        "input_shape": helper.input_shape,
        "n_filters": 12,
        "n_layers_conv": 2,
        "n_layers_unet": 3,
        "kernel_size": (4, 4),
        "dropout": 0.0,
        "batch_norm": False,
        "learning_rate": 0.00005,
        "alpha": 3.0
    }
    model = models.unet(unet_config)

    # Train
    helper.train(model, unet_config)

# Assessment
helper.make_roc_curve()
helper.assess()

# Only write metadata if we trained from scratch
if args.load_weights is None:
    helper.write_metadata()
