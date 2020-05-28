import os, sys

import numpy
import glob
import h5py
import json
import random
import math

import matplotlib.pyplot as plt

import cv2

import pydicom
import nibabel

def load_dcms(files):
    if not len(files) >= 1:
        return None
    
    ct_slices = []
    for file in files:
        f = pydicom.dcmread(file)
        if hasattr(f, 'SliceLocation'): # skip scout views
            ct_slices.append(f)
        else:
            print("[UTILS.PY] load_dcms: found slice that is a scout view (?)")

    ct_slices = sort(ct_slices)
    
    ct_slices_ = []
    ct_slices.reverse() # do I need this?
    for ct_slice in ct_slices:
        ct_slices_.append(ct_slice.pixel_array)
    return numpy.array(ct_slices_).astype(numpy.float64)

def sort(slices):
    slice_locs = [float(s.SliceLocation) for s in slices]
    idx_sorted = numpy.flipud(numpy.argsort(slice_locs)) # flip so we sort head to toe
    return list(numpy.array(slices)[idx_sorted])

def load_nii(file):
    if not os.path.exists(file):
        return None

    label = nibabel.load(file).get_fdata()
    label = numpy.flip(numpy.rot90(label, -1), 1)

    label_ = []
    for i in range(len(label[0,0])):
        label_.append(label[:,:,i])

    return numpy.array(label_).astype(numpy.float64)

def power_of_two(n):
    return math.log2(n).is_integer()

def downsample_images(images, downsample, round = False):
    nPixels = images.shape[-1]

    if (not power_of_two(nPixels) or not power_of_two(downsample) or not (nPixels / downsample).is_integer()):
        print("[UTILS.PY] Original image has %d pixels and you want to downsize to %d pixels, something isn't right." % (nPixels, downsample))
        sys.exit(1)

    print("[UTILS.PY] Original image has %d pixels and we are downsizing to %d pixels" % (nPixels, downsample))

    downsampled_images = []
    for image in images:
        downsampled_images.append(cv2.resize(image, dsize=(downsample, downsample), interpolation=cv2.INTER_CUBIC))

    out = numpy.array(downsampled_images)
    if round:
        return numpy.round(out)
    else:
        return out

def nonzero_entries(array):
    nonzero = []

    array = numpy.array(array)
    for i in range(len(array)):
        flat = array[i].flatten()
        nonzero_idx = numpy.where(flat > 0)[0]
        nonzero += list(flat[nonzero_idx])

    return numpy.array(nonzero)

def scale_image(image): # scales image from 0 to 1
    image = numpy.array(image)

    min_val = numpy.amin(image)
    image += -min_val

    max_val = numpy.amax(image)
    image *= 1./max_val
    return image

fs = 8
def plot_image_and_truth(image, truth, name):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    image_scaled = scale_image(image)
    plt.imshow(image_scaled, cmap = 'gray')
    plt.title("Original CT Scan", fontsize = fs)

    ax = fig.add_subplot(132)
    plt.imshow(truth, cmap = 'gray')
    plt.title("Radiologist Ground Truth", fontsize = fs)

    ax = fig.add_subplot(133)
    blend = image_scaled
    max_val = numpy.amax(blend)
    for i in range(len(truth)):
        for j in range(len(truth)):
            if truth[i][j] == 1:
                blend[i][j] = 1

    plt.imshow(blend, cmap = 'gray')
    plt.title("Original & Truth", fontsize = fs)
    plt.savefig(name)
    plt.close(fig)

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = numpy.linspace(0, 0.8, N+4)
    return mycmap

def make_heatmap(pred, divergent_colormap = False):
    w, h = pred.shape
    x, y = numpy.mgrid[0:w, 0:h]

    if not divergent_colormap:
        cmap = transparent_cmap(plt.cm.cool)
        levels = numpy.linspace(0, 1, 15)
    else:
        cmap = plt.cm.coolwarm
        levels = numpy.linspace(-1, 1, 15)

    return x, y, cmap, levels

def plot_image_truth_and_pred(image, truth, pred, name):
    fig = plt.figure()
    ax = fig.add_subplot(231)

    image_scaled = scale_image(image)
    plt.imshow(image_scaled, cmap = 'gray')
    plt.title("Original", fontsize = fs)

    ax = fig.add_subplot(232)
    plt.imshow(truth, cmap = 'gray')
    plt.title("Ground Truth", fontsize = fs)

    ax = fig.add_subplot(233)
    plt.imshow(pred, cmap = 'gray')
    plt.title("U-Net Prediction", fontsize = fs)

    ax = fig.add_subplot(234)
    plt.imshow(image_scaled, cmap = 'gray')
    x, y, cmap, levels = make_heatmap(truth)
    heatmap = plt.contourf(x, y, truth.transpose(), cmap = cmap, levels = levels)
    cbar = plt.colorbar(heatmap)
    plt.title('Original + Truth', fontsize = fs)

    ax = fig.add_subplot(235)
    plt.imshow(image_scaled, cmap = 'gray')
    x, y, cmap, levels = make_heatmap(pred)
    heatmap = plt.contourf(x, y, pred.transpose(), cmap = cmap, levels = levels)
    cbar = plt.colorbar(heatmap)
    plt.title('Original + Prediction', fontsize = fs)

    ax = fig.add_subplot(236)
    plt.imshow(image_scaled, cmap = 'gray')
    x, y, cmap, levels = make_heatmap(truth - pred, True)
    heatmap = plt.contourf(x, y, (truth - pred).transpose(), cmap = cmap, levels = levels)
    cbar = plt.colorbar(heatmap)
    plt.title('Truth - Prediction', fontsize = fs)

    plt.tight_layout()
    plt.savefig('plots/unet_assessment_%s.pdf' % name)
    plt.close(fig)
    
