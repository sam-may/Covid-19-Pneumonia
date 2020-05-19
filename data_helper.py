import os, sys

import numpy
import glob
import h5py
import json
import random

import utils


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
        for dir in glob.glob(self.input_dir + "/patient11*/"):
            patient = dir.split("/")[-2]
            self.patients[patient] = {
                    "inputs" : glob.glob(dir + "*/"),
                    "X" : [],
                    "y" : []
            }

            for subdir in self.patients[patient]["inputs"]:
                print("[DATA_HELPER] Extracting data from directory %s" % subdir)
                X = utils.load_dcms(glob.glob(subdir + "/*.dcm"))
                y = utils.load_nii(subdir + "/ct_mask_covid_edited.nii")
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
            self.patients[patient]["X"] = utils.downsample_images(data["X"], self.downsample)
            self.patients[patient]["y"] = utils.downsample_images(data["y"], self.downsample, round = True)
 
    def assess(self, N, tag): # make plots of N randomly chosen images
        for i in range(N):
            image, truth = self.select_random_slice()    
            output_img = self.output_dir + "/image_and_truth_%s_%d.pdf" % (tag, i)
            utils.plot_image_and_truth(image, truth, output_img)
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
            self.patients[patient]["X"] *= 1./self.std

        self.metadata["preprocess_scheme"]["mean"] = self.mean
        self.metadata["preprocess_scheme"]["std"] = self.std

    def calculate_mean_and_std(self):
        pixel_values = []
        for patient, data in self.patients.items():
            if len(pixel_values) == 0:
                pixel_values = utils.nonzero_entries(data["X"])
            else:
                pixel_values = numpy.concatenate(pixel_values, utils.nonzero_entries(data["X"]))

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
