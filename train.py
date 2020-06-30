#import setGPU

import os, sys
import argparse

import numpy
import glob
import h5py

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
    default=1
)
args = parser.parse_args()

# Initialize training functions
helper = train_helper.Train_Helper(
    n_extra_slices=args.n_extra_slices,
    input=args.input,
    input_metadata=args.input_metadata,
    tag=args.tag,
    train_frac=0.7,
    fast=False,
    max_epochs=args.max_epochs
)

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
helper.train(model)
helper.make_roc_curve()
helper.assess()
