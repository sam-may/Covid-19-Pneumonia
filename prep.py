import os, sys

import numpy
import glob
import h5py

import matplotlib.pyplot as plt

import cv2

import pydicom
import nibabel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tag", help = "tag to identify this set", type=str, default = "")
parser.add_argument("--downsample", help = "factor n by which to downsample images", type=int, default = 256)
args = parser.parse_args()

import data_helper

input_dir_wuhan = "/public/smay/covid_ct_data/wuhan/"
input_dir_russia = "/public/smay/covid_ct_data/russia/"

helper = data_helper.DataHelper(
    input_dir_wuhan = input_dir_wuhan,
    input_dir_russia = input_dir_russia,
    tag = args.tag,
    downsample  = args.downsample,
)

helper.prep()
