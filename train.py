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

# Initialize training functions
helper = train_helper.TrainHelper()

# If a summary json is supplied, just load weights (don't train)
if helper.summary_json:
    print("[train.py] Loading model from file: %s" % helper.summary_json)
    with open(helper.summary_json, "r") as f_in:
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
if not helper.summary_json:
    helper.write_metadata()
