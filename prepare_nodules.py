import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from helpers.print_helper import print

class NodulesPrepper():
    def __init__(self):
        self.input_hdf5 = "/mnt/data/LungNodules/besser_annotations.hdf5"
        self.bounding_volume_shape = (64, 64, 32)
        self.patient_regex = "^[A-Z][a-z]+_ser_\d+$"
        self.output_hdf5 = h5py.File("features.hdf5", "w")

    def get_annotations(self, ct_scan, patient):
        ct_scan = np.array(ct_scan)
        ct_scan = ct_scan[:,:,:,0]
        # Get COM of nodule to the nearest pixel
        x, y, z = np.nonzero(ct_scan)
        M = float(np.sum(ct_scan)) # 'mass' of nodule
        if M == 0:
            print("No annotations for patient %s" % patient)
            print("--> skipping")
            return
        x_COM = np.sum(x*ct_scan[(x, y, z)])/M
        y_COM = np.sum(y*ct_scan[(x, y, z)])/M
        z_COM = np.sum(z*ct_scan[(x, y, z)])/M
        # Round to nearest pixel
        x_COM = int(round(x_COM))
        y_COM = int(round(y_COM))
        z_COM = int(round(z_COM))
        # Capture nodule
        bound_x, bound_y, bound_z = self.bounding_volume_shape
        scan_x, scan_y, scan_z = ct_scan.shape
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
        bound_volume = ct_scan[x_:_x,y_:_y,z_:_z]
        # Write to output hdf5 file
        self.output_hdf5.create_dataset(patient, data=bound_volume)
        return

    def process(self):
        # Load dataset
        print("Loading data")
        data = h5py.File(self.input_hdf5, "r")
        pattern = re.compile(self.patient_regex)
        patients = [k for k in list(data.keys()) if pattern.match(k)]
        print("Parsing patients")
        # Run annotation search
        for patient in patients:
            print("Processing {}".format(patient))
            self.get_annotations(data.get(patient), patient)
        print("Writing to disk")
        self.output_hdf5.close()
        return

if __name__ == "__main__":
    prepper = NodulesPrepper()
    prepper.process()
