import os, sys

import numpy
import h5py
import random
import glob
import json
from timeit import default_timer as timer

import metrics
import utils

import tensorflow
import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, file, metadata, input_shape, patients, 
                 batch_size=16, verbose=False):
        self.file = file
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

        return X, y

    def get_random_slice(self):
        """
        Produces a random input and label np array consisting of a 
        random CT slice of interest, as well as n slices above and 
        n slices below.

        Returns an input array of shape (M,M,2n+1) where M is the 
        number of pixels in a slice.
        """
        M, M_, n_ = self.input_shape # assuming (M,M,2*n+1)
        n = 2*n_-1
        f = self.file
        # Get random patient and index of slice of interest
        patient = random.choice(self.patients)
        n_slices = len(self.metadata[patient])
        slice_idx = random.randrange(0, N_slices)
        # Label
        y_stack = [f[patient+"_y_"+str(slice_idx)]]
        # Stack n input slices below and n above slice of interest
        X_stack = []
        for idx in range(slice_idx-n, slice_idx+n+1):
            if idx < 0 or idx > n_slices-1:
                # Set slices outside of boundary to zero
                X_stack.append(np.zeros(M, M))
            elif idx == slice_idx:
                X_stack.append(f[patient+"_X_"+str(slice_idx)])
            else:
                X_stack.append(f[patient+"_X_"+str(idx)])

        return np.dstack(X_stack), np.dstack(y_stack)

class Train_Helper():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # General
        self.verbose = kwargs.get('verbose', True)
        self.fast  = kwargs.get('fast', False)
        self.tag = kwargs.get('tag')
        # Data
        self.input = kwargs.get('input')
        self.input_metadata = kwargs.get('input_metadata')
        self.input_shape = (-1, -1, -1) # i.e. (M,M,2n+1)
        self.data_manager = {}
        self.load_data()
        self.n_extra_slices = kwargs.get('n_extra_slices', 0) # i.e. n
        self.n_pixels = self.input_shape[0] # i.e. M
        # Training hyperparameters
        self.train_frac = kwargs.get('train_frac', 0.7)
        self.best_loss = 999999
        self.delta = 0.01 # percent by which loss must improve
        self.early_stopping_rounds = 2
        self.increase_batch = True
        self.decay_learning_rate = False
        self.batch_size = 8
        self.max_batch = 128
        self.max_epochs = kwargs.get('max_epochs', 1)
        # Initialize results object
        self.summary = {
            "input": self.input,
            "train_frac": self.train_frac,
            "config": self.unet_config,
            "predictions": [],
            "metrics": {
                "loss": [], 
                "dice_loss": [], 
                "accuracy": []
            },
            "metrics_train": {
                "loss": [], 
                "dice_loss": [], 
                "accuracy": []
            }
        }


    def load_data(self):
        """
        Load input hdf5 file and set patients array, get pneumonia 
        imbalance for each slice, distribute training and testing 
        data, and store input shape
        """
        self.file = h5py.File(self.input, "r") 
        with open(self.input_metadata, "r") as f_in:
            self.metadata = json.load(f_in)

        self.get_patients()
        self.calculate_pneumonia_imbalance()
        # Calculate number of training/testing slices
        self.n_train = int(self.train_frac*float(len(self.patients)))
        self.n_test  = len(self.patients)-self.n_train
        # Shuffle patients
        patients_shuffle = self.patients
        random.shuffle(patients_shuffle)
        # Distribute training and testing data
        self.patients_train = patients_shuffle[:self.n_train]
        self.patients_test  = patients_shuffle[self.n_train:]
        # Derive/store input shape
        X_ = numpy.array(self.file[self.patients[0] + "_X_0"])
        self.n_pixels = X_.shape[1]
        self.input_shape = (X_.shape[1], X_.shape[2], 
                            2*self.n_extra_slices+1)

    def get_patients(self):
        self.patients = [key for key in self.metadata.keys() 
                             if "patient" in key]
        for pt in self.patients:
            self.data_manager[pt] = []
            for entry in self.metadata[pt]:
                self.data_manager[pt].append({
                    "keys": [entry["X"], entry["y"]], 
                    "n_pneumonia": float(entry["n_pneumonia"])
                })

    def calculate_pneumonia_imbalance(self):
        pneumonia_pixels = 0
        all_pixels = 0
        
        for patient in self.patients:
            for entry in self.metadata[patient]:
                pneumonia_pixels += float(entry["n_pneumonia"])
                all_pixels += float(self.n_pixels**2)

        self.pneumonia_fraction = pneumonia_pixels/all_pixels
        print("[TRAIN_HELPER] Fraction of pixels with pneumonia: %.6f" 
              % self.pneumonia_fraction)

        #self.unet_config["alpha"] = 1.0/self.pneumonia_fraction

    def load_weights(self, weights):
        if self.model is not None:
            self.initialize_model()
        self.model.load_weights(weights)

    def train(self, model):
        self.model = model
        self.train_with_early_stopping()

    # def load_from_file(self, patient):
    #     f_in = h5py.File(self.file_dict[patient]["file"], "r")
    #     X = numpy.array(f_in[self.file_dict[patient]["X"]])
    #     y = numpy.array(f_in[self.file_dict[patient]["y"]])

    #     return X, y

    # def generator(self, patients):
    #     while True:
    #         for patient in patients:
    #             X, y = self.load_features([patient])
    #             N = len(X)
    #             for i in range(N//self.batch_size):
    #                 yield X[(i*self.batch_size):((i+1)*self.batch_size)], 
    #                       y[(i*self.batch_size):((i+1)*self.batch_size)]
    #             yield X[(i+1)*self.batch_size:], 
    #                   y[(i+1)*self.batch_size:]

    # def load_features(self, patients):
    #     X = []
    #     y = []

    #     for patient in patients:
    #         if len(X) == 0:
    #             X, y = self.load_from_file(patient)
    #         else:
    #             X_, y_ = self.load_from_file(patient)
    #             X = numpy.concatenate([X, X_])
    #             y = numpy.concatenate([y, y_])

    #     X = X.reshape([-1, self.n_pixels, self.n_pixels, 1])
    #     y = y.reshape([-1, self.n_pixels, self.n_pixels, 1])

    #     return X, y

    def train_with_early_stopping(self):
        # Save weights to file after ever epoch
        self.weights_file = "weights/"+self.tag+"_weights_{epoch:02d}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_file)
        callbacks_list = [checkpoint]
        # Training loop
        train_more = True
        self.n_epochs = 0
        self.bad_epochs = 0
        while train_more:
            self.train_generator = DataGenerator(
                file=self.file, 
                metadata=self.metadata,
                input_shape=self.input_shape,
                patients=self.patients_train, 
                batch_size=self.batch_size
            )
            self.validation_generator = DataGenerator(
                file=self.file, 
                metadata=self.metadata,
                input_shape=self.input_shape,
                patients=self.patients_test, 
                batch_size=128
            )
            self.n_epochs += 1

            if self.verbose:
                print("[TRAIN_HELPER] On %d-th epoch of training model" 
                      % self.n_epochs)

            results = self.model.fit(
                self.train_generator,
                callbacks=callbacks_list,
                use_multiprocessing=True,
                validation_data=self.validation_generator
            )

            #prediction = self.model.predict(self.validation_generator)
            #self.summary["predictions"].append(prediction)
    
            # TODO: evaluate all metrics with prediction and append to summary

            val_loss = results.history['val_loss'][0]

            for metric in ["loss", "accuracy", "dice_loss"]:
                self.summary["metrics"][metric].append(str(results.history['val_' + metric][0]))
                self.summary["metrics_train"][metric].append(str(results.history[metric][0]))

            percent_change = ((self.best_loss - val_loss)/val_loss)*100.0
            if (val_loss * (1. + self.delta)) < self.best_loss:
                print("[TRAIN_HELPER] Loss improved by %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, self.best_loss, val_loss))
                print("[TRAIN_HELPER] --> continuing for another epoch")
                self.best_loss = val_loss
                self.bad_epochs = 0

            else:
                print("[TRAIN_HELPER] Change in loss was %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, self.best_loss, val_loss)) 
                print("[TRAIN_HELPER] --> incrementing bad epochs by 1")
                self.bad_epochs += 1

            if (self.increase_batch or self.decay_learning_rate) 
                and self.bad_epochs >= 1: 
                # Increase batch size (decay learning rate as well?)
                if self.batch_size * 4 <= self.max_batch:
                    print("[TRAIN_HELPER] --> Increasing batch size from %d -> %d" 
                          % (self.batch_size, self.batch_size*4))
                    print("[TRAIN_HELPER] --> resetting bad epochs to 0")
                    print("[TRAIN_HELPER] --> continuing for another epoch")
                    self.batch_size *= 4
                    self.bad_epochs = 0

            if self.bad_epochs >= self.early_stopping_rounds:
                print("[TRAIN_HELPER] Number of early stopping rounds (%d) without\
                      improvement in loss of at least %.2f percent exceeded" 
                      % (self.early_stopping_rounds, self.delta*100.))
                print("[TRAIN_HELPER] --> stopping training after %d epochs" 
                      % (self.n_epochs))
                train_more = False
        
            if self.max_epochs > 0 and self.n_epochs >= self.max_epochs:
                print("[TRAIN_HELPER] Maximum number of training epochs (%d) reached" 
                      % (self.max_epochs))
                print("[TRAIN_HELPER] --> stopped training")
                train_more = False

        with open("results_%s.json" % self.tag, "w") as f_out:
            json.dump(self.summary, f_out, indent = 4, sort_keys = True) 

    def make_roc_curve(self):
        self.tprs = []
        self.fprs = []
        self.aucs = []

        for i in range(10):
            y = []
            pred = []
            for j in range(3):
                X, y_ = self.validation_generator.__getitem__((i*3)+j)
                pred_ = self.model.predict(X)
                y.append(y_)
                if j == 0:
                    pred = pred_
                else:
                    pred = numpy.concatenate([pred, pred_])

            #X, y = self.validation_generator.__getitem__(i)
            #pred = self.model.predict(X)

            fpr, tpr, auc = utils.calc_auc(y.flatten(), pred.flatten())
            self.fprs.append(fpr)
            self.tprs.append(tpr)
            self.aucs.append(auc)

        tpr_mean = numpy.mean(self.tprs, axis=0)
        tpr_std = numpy.std(self.tprs, axis=0)
        fpr_mean = numpy.mean(self.fprs, axis=0)
        fpr_std = numpy.std(self.fprs, axis=0)
        auc = numpy.mean(self.aucs)
        auc_std = numpy.std(self.aucs)
        
        utils.plot_roc(fpr_mean, fpr_std, tpr_mean, tpr_std, auc, auc_std, "")

    def assess(self): 
        """
        Make plots of (orig|truth|pred)\\(orig+truth|orig+pred|original+(pred-truth))
        """
        X, y = self.validation_generator.__getitem__(0)
        preds = self.model.predict(X, batch_size = 16)
        idx = 0
        for image, truth, pred in zip(X, y, preds):
            image = image.reshape([self.n_pixels, self.n_pixels])
            truth = truth.reshape([self.n_pixels, self.n_pixels])
            pred  = pred.reshape([self.n_pixels, self.n_pixels])       

            utils.plot_image_truth_and_pred(image, truth, pred, "comp_%d" % idx)
            idx += 1
