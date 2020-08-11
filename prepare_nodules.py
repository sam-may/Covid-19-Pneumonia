import os
import re
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from helpers.print_helper import print

class NodulesPrepper():
    def __init__(self, bounding_volume=(64,64,16), patient_regex=""):
        cli = argparse.ArgumentParser()
        cli.add_argument(
            "--mask_hdf5", 
            help="path to annotated CT scan hdf5 file", 
            type=str, 
            default=""
        )
        cli.add_argument(
            "--scan_hdf5", 
            help="path to CT scan hdf5 file", 
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
        self.patient_regex = patient_regex
        # Open HDF5 files
        self.scan_data = h5py.File(self.scan_hdf5, "r")
        self.mask_data = h5py.File(self.mask_hdf5, "r")
        if not self.output_hdf5:
            self.output_hdf5 = self.input_dir+"features.hdf5"
        self.output_data = h5py.File(self.output_hdf5, "w")

    def process_patient(self, patient):
        """
        Isolate annotated lung nodule within a given bounding volume, return 
        save multichannel volume (scan, mask)
        """
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
        scan_x, scan_y, scan_z = ct_mask.shape
        x_ = x_COM - bound_x//2
        _x = x_COM + bound_x//2
        y_ = y_COM - bound_y//2
        _y = y_COM + bound_y//2
        z_ = z_COM - bound_z//2
        _z = z_COM + bound_z//2
        # Check boundaries
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
        bound_scan = ct_scan[x_:_x,y_:_y,z_:_z]
        bound_stack = np.stack([bound_scan, bound_mask], axis=-1)
        return bound_stack

    def process(self):
        """Process data for all patients in annotated dataset"""
        print("Loading patients")
        pattern = re.compile(self.patient_regex)
        patients = [k for k in list(self.mask_data.keys()) if pattern.match(k)]
        masks_in_scans = np.isin(patients, list(self.scan_data.keys()))
        if not np.all(masks_in_scans):
            print("The following patients in %s have no counterpart in %s:" 
                  % (self.mask_hdf5, self.scan_hdf5))
            patients = np.array(patients)
            for patient in patients[~masks_in_scans]:
                print("  - %s" % patient)
            print("--> skipping the patients listed above")
            patients = list(patients[masks_in_scans])
        print("Parsing patients")
        # Run annotation search
        for patient in patients:
            print("Processing {}".format(patient))
            bound_stack = self.process_patient(patient)
            # Write to output hdf5 file
            self.output_data.create_dataset(patient, data=bound_stack)
        print("Wrapping up")
        self.scan_data.close()
        self.mask_data.close()
        self.output_data.close()
        return

if __name__ == "__main__":
    prepper = NodulesPrepper(
        bounding_volume=(64,64,32),
        patient_regex="^[A-Z][a-z]+_ser_\d+$"
    )

    prepper.process()
