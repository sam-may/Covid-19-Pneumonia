import os, sys

import numpy
import glob
import h5py

import utils

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tag", help = "tag to identify this set", type=str, default = "")
parser.add_argument("--input", help = "input hdf5", type=str)
args = parser.parse_args()

import train_helper

helper = train_helper.Train_Helper(
            input       = args.input,
            tag         = args.tag,
            train_frac  = 0.75,
            fast        = False,
)

helper.load_data()
helper.train()
helper.assess()
