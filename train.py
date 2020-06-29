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
parser.add_argument("--tag", help = "tag to identify this set", type=str, default = "")
parser.add_argument("--input", help = "input hdf5", type=str)
parser.add_argument("--input_metadata", help = "json file with metadata", type=str)
args = parser.parse_args()


# Model
unet_config = {
    "n_filters": 12,
    "n_layers_conv": 2,
    "n_layers_unet": 3,
    "kernel_size": 4,
    "dropout": 0.0,
    "batch_norm": False,
    "learning_rate": 0.00005,
    "alpha": 3.0
}

model = models.unet(unet_config)

# Initialize training functions
helper = train_helper.Train_Helper(model=model,
                                   input=args.input,
                                   input_metadata=args.input_metadata,
                                   tag=args.tag,
                                   train_frac=0.7,
                                   fast=False)
# Train
helper.load_data()
helper.train()
helper.make_roc_curve()
helper.assess()
