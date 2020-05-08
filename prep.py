import os, sys

import numpy
import argparse
import h5py
import glob

import matplotlib.pyplot as plt

import pydicom
import nibabel

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--tag", help = "tag to identify this set", type=str)
args = parser.parse_args()

def load_dcms(files):
    ct_slices = []
    for file in files:
        f = pydicom.dcmread(file)
        if hasattr(f, 'SliceLocation'): # skip scout views
            ct_slices.append(f)
    sorted(ct_slices, key=lambda s: s.SliceLocation)
    return ct_slices

def load_nii(file):
    label = nibabel.load(file).get_fdata()
    return label


### Load data ###
f_label = "data/ct_mask_covid_edited.nii"
f_features = glob.glob("data/ser*img*.dcm")

ct_slices = load_dcms(f_features)
label = load_nii(f_label)

for ct_slice in ct_slices:
    print(ct_slice.SliceLocation)

# Plot some slices that have pneumonia
n_slices = len(ct_slices)
for i in range(n_slices):
    has_pneumonia = numpy.sum(label[:,:,i]) > 0
    if has_pneumonia:
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        plt.imshow(ct_slices[i].pixel_array)
        ax2 = fig.add_subplot(122)
        plt.imshow(label[:,:,i])
        plt.savefig("plots/features_and_label_%d.pdf" % i)
        plt.close(fig)

