import numpy
import random
import tensorflow.keras as keras

class DataGenerator2p5D(keras.utils.Sequence):
    def __init__(self, data, metadata, input_shape, patients, batch_size, 
                 verbose=False):
        self.data = data
        self.metadata = metadata
        self.input_shape = input_shape
        self.patients = patients
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_instances = -1

    def __len__(self):
        if self.n_instances > 0:
            return self.n_instances//self.batch_size
        print("[DATA_GENERATOR] Counting total number of instances.")
        self.n_instances = 0
        for patient in self.patients:
            self.n_instances += len(self.metadata[patient])
        print("[DATA_GENERATOR] Found %d instances => %d batches (batch size = %d)" 
              % (self.n_instances, self.n_instances//self.batch_size, self.batch_size))
        return self.n_instances//self.batch_size

    def __getitem__(self, index):
        if self.verbose:
            start = timer()
        X = []
        y = []
        for i in range(self.batch_size):
            X_, y_ = self.get_random_slice()
            X.append(X_)
            y.append(y_)
        if self.verbose:
            end = timer()
            print("[DATA_GENERATOR] Took %.6f seconds to load batch" % (end - start))

        return numpy.array(X), numpy.array(y)

    def get_random_slice(self, patient=None):
        """
        Produces a random input and label np array consisting of a 
        random CT slice of interest, as well as n slices above and 
        n slices below.

        Returns an input array of shape (M,M,2n+1) where M is the 
        number of pixels in a slice.
        """
        M, M_, n_ = self.input_shape # assuming (M,M,2*n+1)
        n = int((n_-1.0)/2.0)
        data = self.data
        # Get random patient and index of slice of interest
        if not patient:
            patient = random.choice(self.patients)
        n_slices = len(self.metadata[patient])
        slice_idx = random.randrange(0, n_slices)
        # Label
        y_stack = [data[patient+"_y_"+str(slice_idx)]]
        # Stack n input slices below and n above slice of interest
        X_stack = []
        for idx in range(slice_idx-n, slice_idx+n+1):
            if idx < 0 or idx > n_slices-1:
                # Set slices outside of boundary to zero
                X_stack.append(numpy.zeros((M, M)))
            elif idx == slice_idx:
                # Slice of interest
                X_stack.append(data[patient+"_X_"+str(slice_idx)])
            else:
                # Extra slice
                X_stack.append(data[patient+"_X_"+str(idx)])
        # Stack output into a multi-channel image
        X = numpy.dstack(X_stack)
        y = numpy.dstack(y_stack)
        return X, y
