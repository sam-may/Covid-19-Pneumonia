import numpy
import random
import tensorflow.keras as keras
from helpers.print_helper import print

class DataGenerator3D(keras.utils.Sequence):
    def __init__(self, data, metadata, input_shape, patients, batch_size, 
                 no_repeats=True, shuffle_patients=False, verbose=False):
        self.data = data
        self.metadata = metadata
        self.input_shape = input_shape
        self.patients = patients
        self.batch_size = batch_size
        self.no_repeats = no_repeats
        self.verbose = verbose
        # Random patient selection
        self.cur_patient = 0
        if shuffle_patients:
            random.shuffle(self.patients)

    def __len__(self):
        return len(self.patients)//self.batch_size

    def __getitem__(self, index):
        X = []
        y = []
        for i in range(self.batch_size):
            X_, y_ = self.get_nodule()
            X.append(X_)
            y.append(y_)

        return numpy.array(X), numpy.array(y)

    def on_epoch_end(self):
        self.cur_patient = 0

    def get_nodule(self, patient=None):
        """
        Retrieve a nodule (M,M,N) bounding volume and binary label
        """
        if not patient:
            if self.no_repeats and self.cur_patient < len(self.patients):
                patient = self.patients[self.cur_patient]
                self.cur_patient += 1
            else:
                patient = random.choice(self.patients)

        X = self.data.get(patient)
        y = self.metadata[patient]["malignant"]

        return X, y
