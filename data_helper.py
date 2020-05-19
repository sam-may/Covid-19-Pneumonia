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

def power_of_two(n):
    return math.log2(n).is_integer() 

def downsample_images(images, downsample):
    nPixels = images.shape[-1]

    if (not power_of_two(nPixels) or not power_of_two(downsample) or not power_of_two(nPixels / downsample)):
        print("[DATA_HELPER] Original image has %d pixels and you want to downsize to %d pixels, something isn't right." % (nPixels, nPixels/downsample))
        sys.exit(1)

    print("[DATA_HELPER] Original image has %d pixels and we are downsizing to %d pixels" % (nPixels, int(nPixels/downsample)))

    downsampled_images = []
    for image in images:
        downsampled_images.append(cv2.resize(image, dsize=(int(nPixels/downsample), int(nPixels/downsample)), interpolation=cv2.INTER_CUBIC))

    return numpy.array(downsampled_images)

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

def plot_image_and_truth(image, truth, name):
    fig = plt.figure()
    ax = fig.add_subplot(131)
    image_scaled = scale_image(image)
    plt.imshow(image_scaled, cmap = 'gray')

    ax = fig.add_subplot(132)
    plt.imshow(truth, cmap = 'gray')

    ax = fig.add_subplot(133)
    #blend = numpy.zeros_like(image)
    blend = image_scaled
    max_val = numpy.amax(blend)
    for i in range(len(truth)):
        for j in range(len(truth)):
            if truth[i][j] == 1:
                blend[i][j] = 1 

    #cv2.addWeighted(image, 1, truth, 1, 0, blend)
    plt.imshow(blend, cmap = 'gray')
    
    plt.savefig(name)
    plt.close(fig)   
    

class Data_Helper():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.input_dir  = kwargs.get('input_dir')
        self.tag        = kwargs.get('tag')
        
        self.downsample = kwargs.get('downsample')

        self.preprocess_method = kwargs.get('preprocess_method', 'z_score')

        self.output_dir = self.input_dir + "/features/" + self.tag + "_" + self.preprocess_method + "_downsample%d/" % self.downsample
        os.system("mkdir -p %s" % self.output_dir)

        self.output = self.output_dir + "features.hdf5"
        self.metadata = {
            "input_dir" : self.input_dir,
            "downsample" : self.downsample,
            "preprocess_scheme" : { "method" : self.preprocess_method },
            "output" : self.output,
            "diagnostic_plots" : []
        }

    def prep(self):
        self.load_data()
        self.assess(10, "raw")
        self.downsample_data()
        self.preprocess()
        self.assess(10, "processed")
        self.write()

    def load_data(self):
        self.patients = {}
        for dir in glob.glob(self.input_dir + "/patient*/"):
            patient = dir.split("/")[-2]
            self.patients[patient] = {
                    "inputs" : glob.glob(dir + "*/"),
                    "X" : [],
                    "y" : []
            }

            for subdir in self.patients[patient]["inputs"]:
                print("[DATA_HELPER] Extracting data from directory %s" % subdir)
                X = load_dcms(glob.glob(subdir + "/*.dcm"))
                y = load_nii(subdir + "/ct_mask_covid_edited.nii")
                self.add_data(patient, X, y)

    def add_data(self, patient, X, y):
        if len(self.patients[patient]["X"]) == 0:
            self.patients[patient]["X"] = X
            self.patients[patient]["y"] = y
        else:
            self.patients[patient]["X"] = numpy.concatenate([self.patients[patient]["X"], X])
            self.patients[patient]["y"] = numpy.concatenate([self.patients[patient]["y"], y])
        
    def downsample_data(self):
        for patient, data in self.patients.items():
            self.patients[patient]["X"] = downsample_images(data["X"], self.downsample)
            self.patients[patient]["y"] = downsample_images(data["y"], self.downsample)
 
    def assess(self, N, tag): # make plots of N randomly chosen images
        for i in range(N):
            image, truth = self.select_random_slice()    
            output_img = self.output_dir + "/image_and_truth_%s_%d.pdf" % (tag, i)
            plot_image_and_truth(image, truth, output_img)
            self.metadata["diagnostic_plots"].append(output_img)

    def select_random_slice(self):
        patient = random.choice(list(self.patients.keys()))
        idx = random.randint(0, len(self.patients[patient]["X"]) - 1)
        return self.patients[patient]["X"][idx], self.patients[patient]["y"][idx]

    def preprocess(self):
        if self.preprocess_method == "none":
            print("[DATA_HELPER] Not applying any preprocessing scheme")
            return
        elif self.preprocess_method == "z_score":
            print("[DATA_HELPER] Applying z score transformation to hounsfeld units")
            self.z_score()
        else:
            print("[DATA_HELPER] Preprocessing method %s is not supported" % self.preprocess_method)
            sys.exit(1)

    def z_score(self):
        self.mean, self.std = self.calculate_mean_and_std()
        print("[DATA_HELPER] Mean: %.3f, Std dev: %.3f" % (self.mean, self.std))
        
        for patient, data in self.patients.items():
            self.patients[patient]["X"] += -self.mean
            self.patients[patient]["y"] *= 1./self.std

        self.metadata["preprocess_scheme"]["mean"] = self.mean
        self.metadata["preprocess_scheme"]["std"] = self.std

    def calculate_mean_and_std(self):
        pixel_values = []
        for patient, data in self.patients.items():
            if len(pixel_values) == 0:
                pixel_values = nonzero_entries(data["X"])
            else:
                pixel_values = numpy.concatenate(pixel_values, nonzero_entries(data["X"]))

            if len(pixel_values) >= 10**6:
                break

        return numpy.mean(pixel_values), numpy.std(pixel_values)

    def write(self):
        f_out = h5py.File(self.output, "w")
        for patient, data in self.patients.items():
            f_out.create_dataset("%s_X" % patient, data = data["X"])
            f_out.create_dataset("%s_y" % patient, data = data["y"])
        f_out.close()

        with open(self.output.replace(".hdf5", ".json"), "w") as f_out:
            json.dump(self.metadata, f_out, indent = 4, sort_keys = True)
