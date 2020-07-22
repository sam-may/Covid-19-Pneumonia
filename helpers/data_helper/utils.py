import os
import sys
import numpy
import math
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
            ct_slices.append(file_data)
        else:
            # skip scout views
            print("[UTILS.PY] load_dcms: found slice that is a scout view (?)")

    ct_slices_ = []

    # Sort slices head-to-toe
    slice_locs = [float(s.SliceLocation) for s in ct_slices]
    idx_sorted = numpy.flipud(numpy.argsort(slice_locs))
    ct_slices = list(numpy.array(ct_slices)[idx_sorted])
    ct_slices.reverse() 
    for ct_slice in ct_slices:
        ct_slices_.append(ct_slice.pixel_array)
    return numpy.array(ct_slices_).astype(numpy.float32) 

def load_nii(nii_file, flip_upside_down=False):
    """
    Decompress *.nii.gz files and retrieve CT slices. Each nii file contains
    every CT slice from a single CT scan.

    Keyword arguments:
    n -- the number of slices above and below slice of interest to include
         as additional channels
    flip_upside_down -- whether or not to flip all of the images upside
         down. nii's from the russia cohort need to be flipped
    """
    if not os.path.exists(nii_file):
        return None

    file_data = nibabel.load(nii_file).get_fdata()

    if flip_upside_down:
        ct_slices = numpy.flip(numpy.rot90(file_data, 0), 1).T # sorted head-to-toe
    else:
        ct_slices = numpy.flip(numpy.rot90(file_data, 2), 1).T # sorted head-to-toe

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
