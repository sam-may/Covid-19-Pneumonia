import os, sys

import numpy
import glob
import h5py
import json
import random

import utils


class DataHelper():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.input_dir_wuhan  = kwargs.get('input_dir_wuhan')
        self.input_dir_russia = kwargs.get('input_dir_russia')
        if self.input_dir_wuhan:
            self.input_dir = self.input_dir_wuhan.replace("wuhan", "")
        elif self.input_dir_russia:
            self.input_dir = self.input_dir_russia.replace("russia", "")
        else:
           raise ValueError('No input directories supplied.')

        self.tag        = kwargs.get('tag')
        
        self.downsample = kwargs.get('downsample')
        # Automatically downsample all images to this size to begin with (some 
        # inputs may be bigger, some will be this size)
        self.min_size   = kwargs.get('min_size', 512) 

        self.preprocess_method = kwargs.get('preprocess_method', 'z_score')

        self.output_dir = (self.input_dir 
                           + "/features/" 
                           + self.tag 
                           + "_" 
                           + self.preprocess_method 
                           + "_downsample%d/" % self.downsample)

        if os.path.exists(self.output_dir):
            print("[DATA_HELPER] Cleaning old contents in directory: %s" 
                  % self.output_dir)
            os.system("rm -rf %s" % self.output_dir)

        os.system("mkdir %s" % self.output_dir)
        os.system("mkdir %s" % self.output_dir + "/images/")

        self.output = self.output_dir + "features.hdf5"

        self.f_out = h5py.File(self.output, "w")

        if os.path.exists(self.output_dir + "features.json"):
            with open(self.output_dir + "features.json", "r") as f_in:
                self.metadata = json.load(f_in)
        else:
            self.metadata = {
                "input_dir_wuhan": self.input_dir_wuhan,
                "input_dir_russia": self.input_dir_russia,
                "downsample": self.downsample,
                "preprocess_scheme": {"method" : self.preprocess_method},
                "output": [],
                "diagnostic_plots": []
            }

    def prep(self):
        self.find_patients()

        ctr = 0
        n_patients = len(self.patients.keys())
        for patient, data in self.patients.items():
            ctr += 1
            print("[DATA_HELPER] On patient %d out of %d (%.1f%%)" 
                  % (ctr, n_patients, (100.*float(ctr))/float(n_patients)))
            
            if os.path.exists(self.output_file(patient)):
                print("[DATA_HELPER] File %s already exists, continuing." 
                      % self.output_file(patient)) 
                continue
            self.load_data(patient)
            self.assess(patient, 10, "raw")
            self.downsample_data(patient)
            self.preprocess(patient)
            self.assess(patient, 10, "processed")
            self.write(patient)
            # Release memory when we're done with the patient
            self.patients[patient] = {} 

        self.f_out.close()
        self.write_metadata()

    def find_patients(self):
        self.patients = {}

        if self.input_dir_wuhan:
            for dir in glob.glob(self.input_dir_wuhan + "/patient*/"):
                patient = dir.split("/")[-2]
                self.patients[patient] = {
                    "inputs": glob.glob(dir+"*/"),
                    "X": [],
                    "y": []
                }

        if self.input_dir_russia:
            for nii in glob.glob(self.input_dir_russia + "/study_*.nii.gz"):
                file_name = nii.split("/")[-1].replace(".nii.gz", "")
                id = int(file_name.replace("study_", ""))
                patient = "russia_patient_%d" % id
                self.patients[patient] = {
                    "features": nii, 
                    "label": (self.input_dir_russia 
                              + "/masks/study_0%d_mask.nii.gz" % id),
                    "X": [],
                    "y": []
                }

    def load_data(self, patient):
        if "russia" in patient:
            X = utils.load_nii(self.patients[patient]["features"])
            y = utils.load_nii(self.patients[patient]["label"])

            self.add_data(patient, X, y)

        else: # wuhan format
            for subdir in self.patients[patient]["inputs"]:
                print("[DATA_HELPER] Extracting data from directory %s" 
                      % subdir)
                X = utils.load_dcms(glob.glob(subdir + "/*.dcm"))
                y = utils.load_nii(subdir + "/ct_mask_covid_edited.nii")
                if X is None or y is None:
                    print("[DATA_HELPER] Did not load features or labels from \
                           directory %s, skipping." % subdir)
                    continue
                if X.shape != y.shape:
                    print("[DATA_HELPER] Input features have shape %s but label \
                           has shape %s -- please check!" 
                          % (str(X.shape), str(y.shape)))
                    #sys.exit(1)
                    continue
                if X.shape[1] < self.min_size:
                    print("[DATA_HELPER] Images are assumed to be at least \
                           %d x %d pixels, but this image is %d x %d pixels \
                           -- please check!" 
                           % (self.min_size, self.min_size, X.shape[1], X.shape[1]))
                    continue
                elif X.shape[1] > self.min_size:
                    print("[DATA_HELPER] Images are assumed to be as small as \
                           %d x %d pixels, and this image is %d x %d pixels, \
                           so we resize it down to be compatible with the rest." 
                          % (self.min_size, self.min_size, X.shape[1], X.shape[1]))
                    X = utils.downsample_images(X, self.min_size)
                    y = utils.downsample_images(y, self.min_size, round = True)

            self.add_data(patient, X, y)

    def add_data(self, patient, X, y):
        p = patient
        if len(self.patients[p]["X"]) == 0:
            self.patients[p]["X"] = X
            self.patients[p]["y"] = y
        else:
            self.patients[p]["X"] = numpy.concatenate([self.patients[p]["X"], X])
            self.patients[p]["y"] = numpy.concatenate([self.patients[p]["y"], y])
        
    def downsample_data(self, patient):
        #for patient, data in self.patients.items():
        self.patients[patient]["X"] = utils.downsample_images(
            self.patients[patient]["X"], 
            self.downsample
        )
        self.patients[patient]["y"] = utils.downsample_images(
            self.patients[patient]["y"], 
            self.downsample, 
            round = True
        )
 
    def assess(self, patient, N, tag):
        """Make plots of N randomly chosen images"""
        output_name = "/images/image_and_truth_%s_%s_%d.pdf" % (tag, patient, i)
        if N > 0:
            for i in range(N):
                image, truth = self.select_random_slice(patient)    
                output_img = self.output_dir + output_name
                utils.plot_image_and_truth(image, truth, output_img)
                self.metadata["diagnostic_plots"].append(output_img)
        else:
            for patient, data in self.patients.items():
                for i in range(len(data["X"])):
                    image, truth = data["X"][i], data["y"][i]
                    output_img = self.output_dir + output_name
                    utils.plot_image_and_truth(image, truth, output_img)
                    self.metadata["diagnostic_plots"].append(output_img)


    def select_random_slice(self, patient=None):
        if patient is None:
            patient = random.choice(list(self.patients.keys()))
        idx = random.randint(0, len(self.patients[patient]["X"]) - 1)
        return self.patients[patient]["X"][idx], self.patients[patient]["y"][idx]

    def preprocess(self, patient):
        if self.preprocess_method == "none":
            print("[DATA_HELPER] Not applying any preprocessing scheme")
            return
        elif self.preprocess_method == "z_score":
            print("[DATA_HELPER] Applying z score transform to hounsfeld units")
            self.z_score(patient)
        else:
            print("[DATA_HELPER] Preprocessing method %s is not supported" 
                  % self.preprocess_method)
            sys.exit(1)

    def z_score(self, patient):
        if not ("mean" in self.metadata["preprocess_scheme"].keys() 
                or "std" in self.metadata["preprocess_scheme"].keys()):
            mean, std = self.calculate_mean_and_std()
            print("[DATA_HELPER] Mean: %.3f, Std dev: %.3f" % (mean, std))
            self.metadata["preprocess_scheme"]["mean"] = mean
            self.metadata["preprocess_scheme"]["std"] = std
        
        self.patients[patient]["X"] += -self.metadata["preprocess_scheme"]["mean"]
        self.patients[patient]["X"] *= 1./self.metadata["preprocess_scheme"]["std"]

    def calculate_mean_and_std(self):
        pixel_values = []
        for patient, data in self.patients.items():
            if len(pixel_values) == 0:
                pixel_values = utils.nonzero_entries(data["X"])
            else:
                pixel_values = numpy.concatenate(
                    pixel_values, 
                    utils.nonzero_entries(data["X"])
                )

            if len(pixel_values) >= 10**6:
                break

        mean = numpy.mean(pixel_values.astype(numpy.float64))
        std = numpy.std(pixel_values.astype(numpy.float64))

        return mean, std

    def output_file(self, patient):
        return self.output.replace("features.hdf5", "features_%s.hdf5" % patient)

    def write(self, patient):
        self.metadata[patient] = []
        idx = 0
        for X, y in zip(self.patients[patient]["X"], self.patients[patient]["y"]):
            dset_X = "%s_X_%d" % (patient, idx)
            dset_y = "%s_y_%d" % (patient, idx)

            self.f_out.create_dataset(dset_X, data = X)
            self.f_out.create_dataset(dset_y, data = y)

            n_pneumonia = str(numpy.sum(y))
            self.metadata[patient].append({
                "X": dset_X, 
                "y": dset_y, 
                "n_pneumonia": n_pneumonia
            }) 

            idx += 1

        self.write_metadata()

    def write_metadata(self):
        with open(self.output.replace(".hdf5", ".json"), "w") as f_out:
            json.dump(self.metadata, f_out, indent = 4, sort_keys = True)
