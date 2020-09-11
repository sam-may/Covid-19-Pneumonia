import numpy
import random
import tensorflow.keras as keras
from scipy import ndimage
from helpers.print_helper import print

class DataGenerator3D(keras.utils.Sequence):
    def __init__(self, data, metadata, input_shape, patients, batch_size, 
                 input_reshape=None, no_repeats=True, extra_features=[],
                 do_rotations=False, verbose=False):
        self.data = data
        self.metadata = metadata
        self.input_shape = input_shape
        self.input_reshape = input_reshape if input_reshape else input_shape
        self.patients = patients
        self.batch_size = batch_size
        self.no_repeats = no_repeats
        self.extra_features = extra_features
        self.do_rotations = do_rotations
        self.verbose = verbose
        # Random patient selection
        self.cur_patient = 0

    def __len__(self):
        return len(self.patients)//self.batch_size

    def __getitem__(self, index):
        X = []
        x = []
        y = []
        for i in range(self.batch_size):
            X_, x_, y_ = self.get_nodule()
            X.append(X_)
            x.append(x_)
            y.append(y_)

        if self.extra_features:
            return [numpy.array(X), numpy.array(x)], numpy.array(y)
        else:
            return numpy.array(X), numpy.array(y)

    def on_epoch_end(self):
        self.cur_patient = 0

    def augment(self, X):
        ct_scan = X[:,:,:,0]
        ct_mask = X[:,:,:,1]
        # Perform random rotations
        if self.do_rotations:
            azimuthal_rotation = random.randint(0,359)
            ct_mask = ndimage.rotate(
                ct_mask, 
                azimuthal_rotation, 
                reshape=False,
                order=1
            )
            ct_scan = ndimage.rotate(
                ct_scan, 
                azimuthal_rotation, 
                reshape=False,
                order=1
            )
        # Trim bounding box to proper input dimensions
        bound_x, bound_y, bound_z, _ = self.input_shape
        target_x, target_y, target_z = self.input_reshape
        x_ = (bound_x - target_x)//2
        y_ = (bound_y - target_y)//2
        z_ = (bound_z - target_z)//2
        _x = target_x + x_
        _y = target_y + y_
        _z = target_z + z_
        ct_scan = ct_scan[x_:_x, y_:_y, z_:_z]
        ct_mask = ct_mask[x_:_x, y_:_y, z_:_z]
        X = numpy.stack([ct_scan, ct_mask], axis=-1)
        return X

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
        # Input
        X = self.augment(self.data.get(patient))
        # Extra features
        x = []
        if self.extra_features:
            for feature_name in self.extra_features:
                feature = self.metadata[patient][feature_name]
                if type(feature) == list:
                    x += feature
                else:
                    x += [feature]
        # Label
        y = self.metadata[patient]["malignant"]

        return X, x, y

