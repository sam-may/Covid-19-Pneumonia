import os, sys

import numpy
import glob
import h5py
import json
import random
import math

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import cv2

import pydicom
import nibabel

def load_dcms(dcm_files):
    """
    Load DICOM (DCM) files and retrieve CT slices. Each DCM file contains 
    just one CT slice from a given CT scan.

    Keyword arguments:
    n -- the number of slices above and below slice of interest to include
         as additional channels
    """
    if not len(dcm_files) >= 1:
        return None
    
    ct_slices = []
    for dcm_file in dcm_files:
        file_data = pydicom.dcmread(dcm_file)
        if hasattr(file_data, 'SliceLocation'):
            ct_slices.append(file_data.pixel_array)
        else:
            # skip scout views
            print("[UTILS.PY] load_dcms: found slice that is a scout view (?)")

    # Sort slices head-to-toe
    slice_locs = [float(s.SliceLocation) for s in ct_slices]
    idx_sorted = numpy.flipud(numpy.argsort(slice_locs))
    ct_slices = list(numpy.array(ct_slices)[idx_sorted])
    ct_slices.reverse() # do I need this?

    return ct_slices

def load_nii(nii_file):
    """
    Decompress *.nii.gz files and retrieve CT slices. Each nii file contains
    every CT slice from a single CT scan.

    Keyword arguments:
    n -- the number of slices above and below slice of interest to include
         as additional channels
    """
    if not os.path.exists(nii_file):
        return None

    file_data = nibabel.load(nii_file).get_fdata()
    ct_slices = numpy.flip(numpy.rot90(file_data, -1), 1).T # sorted head-to-toe

    return ct_slices

def is_power_of_two(n):
    """Returns True if n is a power of two"""
    return math.log2(n).is_integer()

def downsample_images(images, downsample, round=False):
    n_pixels = images.shape[-1]

    if (not is_power_of_two(n_pixels) or not is_power_of_two(downsample) 
        or not (n_pixels/downsample).is_integer()):
        print("[UTILS.PY] Original image has %d pixels and you want to \
               downsize to %d something isn't right." % (n_pixels, downsample))
        sys.exit(1)

    print("[UTILS.PY] Original image has %d pixels, downsizing to %d pixels" 
          % (n_pixels, downsample))

    downsampled_images = []
    for image in images:
        downsampled_image = cv2.resize(
            image, 
            dsize=(downsample, downsample), 
            interpolation=cv2.INTER_CUBIC
        )
        downsampled_images.append(downsampled_image)

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

def scale_image(image):
    """Normalize image pixel values to range from 0 to 1"""
    image = numpy.array(image)

    min_val = numpy.amin(image)
    image += -min_val

    max_val = numpy.amax(image)
    image *= 1./max_val
    return image

def plot_image_and_truth(image, truth, name, fs=6):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    image_scaled = scale_image(image)
    plt.imshow(image_scaled, cmap='gray')
    plt.title("Original CT Scan", fontsize=fs)

    ax = fig.add_subplot(132)
    plt.imshow(truth, cmap='gray')
    plt.title("Radiologist Ground Truth", fontsize=fs)

    ax = fig.add_subplot(133)
    blend = image_scaled
    max_val = numpy.amax(blend)
    for i in range(len(truth)):
        for j in range(len(truth)):
            if truth[i][j] == 1:
                blend[i][j] = 1

    plt.imshow(blend, cmap='gray')
    plt.title("Original & Truth", fontsize=fs)
    plt.savefig(name)
    plt.close(fig)

def transparent_cmap(cmap, N=255):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = numpy.linspace(0, 0.8, N+4)
    return mycmap

def make_heatmap(pred, divergent_colormap=False):
    w, h = pred.shape
    x, y = numpy.mgrid[0:w, 0:h]

    if not divergent_colormap:
        cmap = transparent_cmap(plt.cm.cool)
        levels = numpy.linspace(0, 1, 15)
    else:
        cmap = plt.cm.bwr
        levels = numpy.linspace(-1, 1, 15)

    return x, y, cmap, levels

def plot_image_truth_and_pred(image, truth, pred, name, fs=6):
    fig = plt.figure()
    ax = fig.add_subplot(231)

    image_scaled = scale_image(image)
    plt.imshow(image_scaled, cmap='gray')
    plt.title("Original", fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(232)
    plt.imshow(truth, cmap='gray')
    plt.title("Ground Truth", fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(233)
    plt.imshow(pred, cmap='gray')
    plt.title("U-Net Prediction", fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(234)
    plt.imshow(image_scaled, cmap='gray')
    x, y, cmap, levels = make_heatmap(truth)
    heatmap = plt.contourf(x, y, truth.transpose(), cmap=cmap, levels=levels)
    plt.title('Original + Truth', fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(235)
    plt.imshow(image_scaled, cmap='gray')
    x, y, cmap, levels = make_heatmap(pred)
    heatmap = plt.contourf(x, y, pred.transpose(), cmap=cmap, levels=levels)
    plt.title('Original + Prediction', fontsize=fs)
    plt.axis('off')

    ax = fig.add_subplot(236)
    plt.imshow(image_scaled, cmap='gray')
    x, y, cmap, levels = make_heatmap(truth - pred, True)
    corrmap = plt.contourf(
        x, 
        y, 
        (truth - pred).transpose(), 
        cmap=cmap, 
        levels=levels
    )
    plt.title('Truth - Prediction', fontsize=fs)
    plt.axis('off')


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.55, 0.05, 0.35])
    cbar = fig.colorbar(heatmap, cax=cbar_ax)
    cbar.ax.set_ylabel(
        'Pneumonia Score', 
        rotation=270, 
        labelpad=15, 
        fontsize=fs
    ) 
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.35])
    cbar = fig.colorbar(corrmap, cax=cbar_ax)
    cbar.ax.set_ylabel(
        'Truth - Prediction', 
        rotation=270, 
        labelpad=15, 
        fontsize=fs
    )
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    #plt.tight_layout()
    plt.savefig('plots/unet_assessment_%s.pdf' % name, bbox_inches='tight')
    plt.close(fig)

def plot_roc(fpr_mean, tpr_mean, tpr_std, auc, auc_std, tag):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.yaxis.set_ticks_position('both')
    ax1.grid(True)
    ax1.plot(
        fpr_mean, 
        tpr_mean,
        color='blue', 
        label="U-Net [AUC: %.3f +/- %.3f]" % (auc, auc_std)
    )
    ax1.fill_between(
        fpr_mean,
        tpr_mean - (tpr_std/2.),
        tpr_mean + (tpr_std/2.),
        color='blue',
        alpha=0.25, label=r'$\pm 1\sigma$'
    )
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig("plots/roc_comparison_%s.pdf" % tag)
    plt.close(fig)

def interpolated_roc(y, pred, n_interp):
    """
    Calculate fpr and tpr in arrays of length n_interp.
    """ 
    
    fpr, tpr, thresh = roc_curve(y, pred)
    fpr_interp = numpy.linspace(0, 1, n_interp)
    tpr_interp = numpy.interp(fpr_interp, fpr, tpr)

    return fpr_interp, tpr_interp

def calc_auc(y, pred, n_interp=1000, n_bootstrap = -1):
    """
    Calculate fpr, tpr, and auc from label y and prediction pred.
    Returns fixed length fpr/tpr arrays of size n_interp.
    If n_bootstrap > 0, also returns uncertainties estimated through 
    n_bootstrap bootstrap resamples.
    """

    fpr, tpr = interpolated_roc(y, pred, n_interp)
    auc_ = auc(fpr, tpr)

    if n_bootstrap > 0:
        n_points = len(y)
        bootstrap_aucs = [] 
        bootstrap_tprs = []
        for i in range(n_bootstrap):
            bootstrap_indices = numpy.random.randint(0, n_points, n_points)
            bootstrap_label = y[bootstrap_indices]
            bootstrap_pred = pred[bootstrap_indices]
            
            fpr_b, tpr_b = interpolated_roc(bootstrap_label, bootstrap_pred, n_interp)
            auc_b = auc(fpr_b, tpr_b)

            bootstrap_aucs.append(auc_b)
            bootstrap_tprs.append(tpr_b)

        auc_unc = numpy.std(bootstrap_aucs)
        tpr_unc = numpy.std(bootstrap_tprs, axis=0)

    else:
        auc_unc = -1
        tpr_unc = -1

    return fpr, tpr, auc_, auc_unc, tpr_unc
