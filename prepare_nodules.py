import os
import re
import h5py
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from helpers.print_helper import print

class NodulesPrepper():
    def __init__(self, bounding_volume=(20,20,20), patient_regex=""):
        cli = argparse.ArgumentParser()
        cli.add_argument(
            "--mask_hdf5", 
            required=True,
            help="path to annotated CT scan hdf5 file", 
            type=str, 
            default=""
        )
        cli.add_argument(
            "--scan_hdf5", 
            required=True,
            help="path to CT scan hdf5 file", 
            type=str, 
            default=""
        )
        cli.add_argument(
            "--dicom_json", 
            required=True,
            help="path to processed JSON dump of DICOM file for CT scans", 
            type=str, 
            default=""
        )
        cli.add_argument(
            "--output_hdf5", 
            help="(optional) path to output hdf5 file", 
            type=str, 
            default=""
        )
        # Load CLI args into namespace
        cli.parse_args(namespace=self)
        self.input_dir = os.path.dirname(self.scan_hdf5)+"/"
        # Load kwargs
        self.bounding_volume = bounding_volume
        self.pattern = re.compile(patient_regex)
        # Open HDF5 files
        self.scan_data = h5py.File(self.scan_hdf5, "r")
        self.mask_data = h5py.File(self.mask_hdf5, "r")
        self.patients = [k for k in list(self.mask_data.keys()) 
                         if self.pattern.match(k)]
        if not self.output_hdf5:
            self.output_dir = self.input_dir
        else:
            self.output_dir = os.path.dirname(self.output_hdf5)
            if self.output_dir != "":
                self.output_dir += "/"
        self.output_hdf5 = self.output_dir+"features.hdf5"
        self.metadata_json = self.output_dir+"metadata.json"
        self.output_data = h5py.File(self.output_hdf5, "w")
        self.metadata = {}
        # Open DICOM JSON
        with open(self.dicom_json, "r") as f_in:
            self.dicom_data = json.load(f_in)
        # Mean and std for z-score normalization
        self.mean, self.std = self.calc_mean_and_std()

    def calc_mean_and_std(self):
        """Calculate mean and std dev of ALL pixels for z-score norm"""
        print("Calculating mean and std of all pixels")
        pixel_values = []
        for patient in self.patients:
            ct_scan = self.scan_data.get(patient)
            # Get random slices from scan
            for _ in range(5):
                random_z = random.randint(0, ct_scan.shape[-1]-1)
                ct_slice = ct_scan[:,:,random_z]
                # Get all nonzero pixels
                slice_pixels = ct_slice[np.nonzero(ct_slice)].flatten()
                pixel_values = np.concatenate((pixel_values, slice_pixels))
                # Arbitrary cap
                if len(pixel_values) >= 10**6:
                    break
        mean = np.mean(pixel_values.astype(np.float64))
        std = np.std(pixel_values.astype(np.float64))
        print("Result: mean = %.3f, std = %.3f" % (mean, std))
        return mean, std

    def add_metadata(self, patient, patient_metadata):
        if patient in self.metadata.keys():
            self.metadata[patient].update(patient_metadata)
        else:
            self.metadata[patient] = patient_metadata
        return

    def write_metadata(self):
        print("Writing metadata to %s" % self.metadata_json)
        with open(self.metadata_json, "w") as f_out:
            json.dump(self.metadata, f_out, indent=4)
        return

    def process_patient(self, patient):
        """
        Isolate annotated lung nodule within a given bounding volume, return 
        save multichannel volume (scan, mask)
        """
        # DICOM metadata
        if patient not in self.dicom_data.keys():
            print("No DICOM data for patient %s" % patient)
            print("--> skipping")
            return
        patient_dicom = self.dicom_data[patient]
        z_spacing = 1.0
        if "SpacingBetweenSlices" in patient_dicom.keys():
            z_spacing = patient_dicom["SpacingBetweenSlices"] # in mm
        elif "SliceThickness" in patient_dicom.keys():
            z_spacing = patient_dicom["SliceThickness"]/2.0 # in mm
        else:
            print("No slice thickness data for patient %s" % patient)
            print("--> skipping")
            return
        x_spacing, y_spacing = patient_dicom["PixelSpacing"] # in mm
        # Load scan h5py dataset object
        ct_scan = self.scan_data.get(patient)
        # Load mask directly to memory
        ct_mask = np.array(self.mask_data.get(patient))
        ct_mask = ct_mask[:,:,:,0]
        # Get COM of nodule to the nearest pixel
        M = float(np.sum(ct_mask)) # 'mass' of nodule
        if M == 0:
            print("No annotations for patient %s" % patient)
            print("--> skipping")
            return
        x, y, z = np.nonzero(ct_mask)
        x_COM = np.sum(x*ct_mask[(x, y, z)])/M
        y_COM = np.sum(y*ct_mask[(x, y, z)])/M
        z_COM = np.sum(z*ct_mask[(x, y, z)])/M
        # Round to nearest pixel
        x_COM = int(round(x_COM))
        y_COM = int(round(y_COM))
        z_COM = int(round(z_COM))
        # Get bounding volume edges
        bound_x, bound_y, bound_z = self.bounding_volume
        # Calculate bound volume edges
        x_ = x_COM - bound_x//2
        _x = x_COM + bound_x//2
        y_ = y_COM - bound_y//2
        _y = y_COM + bound_y//2
        z_ = z_COM - bound_z//2
        _z = z_COM + bound_z//2
        # Check scan boundaries
        scan_x, scan_y, scan_z = ct_mask.shape
        if x_ < 0 or y_ < 0 or z_ < 0:
            print("Bounding box outside of CT volume")
            print("--> skipping")
            return
        if _x > scan_x or _y > scan_y or _z > scan_z:
            print("Bounding box outside of CT volume")
            print("--> skipping")
            return
        # Fill bounding volume
        bound_mask = ct_mask[x_:_x,y_:_y,z_:_z]
        bound_scan = ct_scan[x_:_x,y_:_y,z_:_z].astype(np.float64)
        # Apply z-score norm to bound CT scan volume
        bound_scan -= self.mean
        bound_scan *= 1./self.std
        # Stack inputs
        bound_stack = np.stack([bound_scan, bound_mask], axis=-1)
        # Volumetric metadata
        bound_M = float(np.sum(bound_mask))
        nodule_volume = np.sum(ct_mask > 0)*x_spacing*y_spacing*z_spacing
        patient_metadata = {
            "center_of_mass": [
                x_COM*x_spacing, 
                y_COM*y_spacing, 
                z_COM*z_spacing
            ],
            "nodule_volume": nodule_volume,
            "exceeds_volume": int(M > bound_M)
        }
        if M < bound_M:
            print("uh... what? this is %s" % patient)
        self.add_metadata(patient, patient_metadata)
        return bound_stack

    def process(self):
        """Process data for all patients in annotated dataset"""
        print("Loading patients")
        masks_in_scans = np.isin(self.patients, list(self.scan_data.keys()))
        if not np.all(masks_in_scans):
            print("The following patients in %s have no counterpart in %s:" 
                  % (self.mask_hdf5, self.scan_hdf5))
            self.patients = np.array(self.patients)
            for patient in self.patients[~masks_in_scans]:
                print("  - %s" % patient)
            print("--> skipping the patients listed above")
            self.patients = list(self.patients[masks_in_scans])
        print("Parsing patients")
        # Run annotation search
        for patient in self.patients:
            print("Processing {}".format(patient))
            bound_stack = self.process_patient(patient)
            # Write to output hdf5 file
            if np.any(bound_stack):
                self.output_data.create_dataset(patient, data=bound_stack)
        print("Wrapping up")
        self.scan_data.close()
        self.mask_data.close()
        self.output_data.close()
        self.write_metadata()
        return

    def add_labels(self, benign_txt, scc_txt, adeno_txt, existing_json=None):
        """Add biopsy labels to patient metadata"""
        if existing_json:
            with open(existing_json, "r") as f_in:
                self.metadata = json.load(f_in)
        # Get labels
        print("Loading labels")
        benign = []
        with open(benign_txt) as f_in:
            for line in f_in.readlines():
                benign.append(line.split("\n")[0])
        scc = []
        with open(scc_txt) as f_in:
            for line in f_in.readlines():
                scc.append(line.split("\n")[0])
        adeno = []
        with open(adeno_txt) as f_in:
            for line in f_in.readlines():
                adeno.append(line.split("\n")[0])
        # Get unique list of labeled patients
        labeled_patients = list(set(benign+scc+adeno))
        # Fill metadata
        for patient_id in self.patients:
            patient_metadata = {}
            patient = patient_id.split("_ser_")[0]
            if patient not in labeled_patients:
                print("%s not in labeled patients" % patient)
                print("--> skipping")
                continue
            patient_metadata["benign"] = int(patient in benign)
            patient_metadata["malignant"] = int(not patient in benign)
            # Squamous cell carcinoma (malignant)
            patient_metadata["scc"] = int(patient in scc)
            # Adenocarcinoma (malignant)
            patient_metadata["adeno"] = int(patient in adeno)
            # Save to metadata
            self.add_metadata(patient_id, patient_metadata)

        self.write_metadata()
        return

    def normalize_metadata(self, name, existing_json=None):
        print("Adding normalized '%s' values" % name)
        if existing_json:
            with open(existing_json, "r") as f_in:
                self.metadata = json.load(f_in)
        # Get mean and std of metadata field
        values = []
        componentwise = False
        for patient, data in self.metadata.items():
            if name not in data.keys():
                continue
            if len(values) == 0:
                componentwise = (len(np.array(data[name]).shape) == 1)
            values.append(data[name])
        values = np.array(values)
        means = []
        stds = []
        if componentwise:
            for c in range(values.shape[-1]):
                means.append(np.mean(values[:,c]))
                stds.append(np.std(values[:,c]))
        else:
            means.append(np.mean(values.flatten()))
            stds.append(np.std(values.flatten()))
        for i, mean in enumerate(means):
            std = stds[i]
            for patient, data in self.metadata.items():
                if name not in data.keys():
                    continue
                if componentwise:
                    new_data = (data[name][i] - mean)/std
                    if i == 0:
                        self.metadata[patient][name+"_norm"] = [new_data]
                    else:
                        self.metadata[patient][name+"_norm"].append(new_data)
                else:
                    self.metadata[patient][name+"_norm"] = (data[name] - mean)/std
        self.write_metadata()
        return
            

if __name__ == "__main__":
    prepper = NodulesPrepper(
        bounding_volume=(64,64,32),
        patient_regex="^[A-Z][a-z]+_ser_\d+$"
    )
    # Run data pre-processing
    prepper.process()
    prepper.add_labels(
        benign_txt="benign.txt",
        scc_txt="malignant-SCC.txt",
        adeno_txt="malignant-adeno.txt"
    )
    prepper.normalize_metadata("center_of_mass")
    prepper.normalize_metadata("nodule_volume")
