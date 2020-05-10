import os, sys

import numpy
import argparse
import h5py
import glob

import matplotlib.pyplot as plt

import cv2

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
    ct_slices_ = []
    for ct_slice in ct_slices:
        ct_slices_.append(ct_slice.pixel_array)
    return numpy.array(ct_slices_).astype(numpy.float64)

def load_nii(file):
    label = nibabel.load(file).get_fdata()
    label = numpy.flip(numpy.rot90(label, -1), 1)

    label_ = []
    for i in range(len(label[0,0])):
        label_.append(label[:,:,i])

    return numpy.array(label_).astype(numpy.float64)


### Load data ###
f_label = "data/ct_mask_covid_edited.nii"
f_features = glob.glob("data/ser*img*.dcm")

ct_slices = load_dcms(f_features)
label = load_nii(f_label)

def make_pixel_value_histogram(ct_slices, tag, subsample = 0.05):
    n_subsample = int(subsample * float(len(ct_slices)))
    idx = numpy.random.randint(0, len(ct_slices), n_subsample)
    
    values = ct_slices[idx].flatten()
    print(tag, len(values))

    hist, bins = numpy.histogram(values)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(values, bins = bins)
    plt.savefig("pixel_value_hist_%s.pdf" % tag)

def z_transform(img_array):
    nonzero_component = img_array[numpy.nonzero(img_array)].flatten()

    print(len(img_array.flatten()), len(nonzero_component))

    mean = numpy.mean(nonzero_component)
    std = numpy.std(nonzero_component)

    img_array_ = img_array.astype(numpy.float64)

    img_array_ += -mean
    img_array_ *= 1./std

    return img_array_

def ln_img(img_array):
    img_array_ = img_array.astype(numpy.float64)
    img_array_ = numpy.log(img_array, out=numpy.zeros_like(img_array_), where=(img_array_ != 0))
    return img_array_

def reduce_images(img_array, n_pixels):
    reduced_imgs = []
    for img in img_array:
        reduced_imgs.append(cv2.resize(img, dsize=(n_pixels,n_pixels), interpolation=cv2.INTER_CUBIC))

    return numpy.array(reduced_imgs)

def plot_images(ct_slices, label):
    n_slices = len(ct_slices)
    for i in range(n_slices):
        has_pneumonia = numpy.sum(label[i]) > 0
        if has_pneumonia:
            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            plt.imshow(ct_slices[i])

            ax2 = fig.add_subplot(132)
            plt.imshow(label[i])

            ax3 = fig.add_subplot(133)
            blend = numpy.zeros_like(ct_slices[i])
            cv2.addWeighted(ct_slices[i], 1, label[i], 1000, 0, blend)
            plt.imshow(blend)

            plt.savefig("plots/features_and_label_%d.pdf" % i)
            plt.close(fig)


ct_slices = reduce_images(ct_slices, 128)
label = reduce_images(label, 128)

plot_images(ct_slices,label)

ct_slices = z_transform(ct_slices)

label_simple = []
for img in label:
    if numpy.sum(img) > 0:
        label_simple.append(1)
    else:
        label_simple.append(0)

label_simple = numpy.array(label_simple)

output_file = "covid_data.hdf5"

f_out = h5py.File(output_file, "w")
f_out.create_dataset("features", data = ct_slices)
f_out.create_dataset("labels", data = label)
f_out.create_dataset("label_simple", data = label_simple)
f_out.close()
