import os, sys

import numpy
import h5py
import random
import glob
import json
from timeit import default_timer as timer

import models
import utils

import tensorflow
import tensorflow.keras as keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, file, n_pixels, patients, metadata, batch_size = 16):
        self.batch_size = batch_size
        self.metadata = metadata
        self.patients = patients
        self.file = file
        self.n_pixels = n_pixels

        self.n_instances = -1

    def __len__(self):
        if self.n_instances > 0:
            return self.n_instances // self.batch_size

        print("[DATA_GENERATOR] Counting total number of instances.")

        self.n_instances = 0
        for patient in self.patients:
            self.n_instances += len(self.metadata[patient])

        print("[DATA_GENERATOR] Found %d total instances => %d total batches (batch size of %d)" % (self.n_instances, self.n_instances // self.batch_size, self.batch_size))
        return self.n_instances // self.batch_size

    def get_random_slice(self):
        pt, idx = self.get_random_patient_and_idx()

        X = pt + "_X_" + str(idx)
        y = pt + "_y_" + str(idx)

        return X, y

    def get_random_patient_and_idx(self):
        pt = random.choice(self.patients)
        idx = random.randrange(0, len(self.metadata[pt]))

        return pt, idx

    def __getitem__(self, index):
        start = timer()
        for i in range(self.batch_size):
            X_, y_ = self.get_random_slice()

            if i == 0:
                X = numpy.array(self.file[X_])
                y = numpy.array(self.file[y_])
            else:
                X = numpy.concatenate([X, numpy.array(self.file[X_])])
                y = numpy.concatenate([y, numpy.array(self.file[y_])])

        X = X.reshape([-1, self.n_pixels, self.n_pixels, 1])
        y = y.reshape([-1, self.n_pixels, self.n_pixels, 1])

        end = timer()
        print("[DATA_GENERATOR] Took %.6f seconds to load batch" % (end - start))

        return X, y

class Train_Helper():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.fast           = kwargs.get('fast', False)

        self.input          = kwargs.get('input')
        self.input_metadata = kwargs.get('input_metadata')
        self.tag            = kwargs.get('tag')
        self.verbose        = kwargs.get('verbose', True)

        self.train_frac     = kwargs.get('train_frac', 0.7)

        self.unet_config    = kwargs.get('unet_config', {
                                            "n_filters" : 16,
                                            "n_layers_conv" : 1,
                                            "n_layers_unet" : 3,
                                            "kernel_size" : 3,
                                            "dropout" : 0.0,
                                            "batch_norm" : False,
                                            "learning_rate" : 0.00005,
                                        })

        self.best_loss = 999999
        self.delta = 0.01 # percent by which loss must improve to be considered an improvement
        self.early_stopping_rounds = 2
    
        self.increase_batch = False
        self.decay_learning_rate = False
        self.batch_size = 16
        self.max_batch  = 16
        self.max_epochs = 10

        self.n_assess = 25
        self.n_pixels = -1

        # initialize places to store results
        self.summary = {
                "input"         : self.input,
                "train_frac"    : self.train_frac,
                "config"        : self.unet_config,
                "predictions"   : [],
                "metrics"       : { "binary_crossentropy" : [], "dice" : [] },
                "metrics_train" : { "binary_crossentropy" : [], "dice" : [] },
        }

    def load_data(self):
        self.file = h5py.File(self.input, "r") 
        with open(self.input_metadata, "r") as f_in:
            self.metadata = json.load(f_in)

        self.get_patients()

        self.n_train = int(self.train_frac * float(len(self.patients)))
        self.n_test  = len(self.patients) - self.n_train

        patients_shuffle = self.patients
        random.shuffle(patients_shuffle)

        self.patients_train = patients_shuffle[:self.n_train]
        self.patients_test  = patients_shuffle[self.n_train:]

    def load_weights(self, weights):
        if self.model is not None:
            self.initialize_model()
        self.model.load_weights(weights)

    def train(self):
        self.initialize_model()
        self.train_with_early_stopping()

    def generator(self, patients):
        while True:
            for patient in patients:
                X, y = self.load_features([patient])
                N = len(X)
                for i in range(N//self.batch_size):
                    yield X[ (i*self.batch_size) : ((i+1)*self.batch_size)], y[ (i*self.batch_size) : ((i+1)*self.batch_size) ]
                yield X[(i+1)*self.batch_size:], y[(i+1)*self.batch_size:]

    def train_with_early_stopping(self):
        self.weights_file = "weights/" + self.tag + "_weights_{epoch:02d}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_file) # save after every epoch
        callbacks_list = [checkpoint]

        train_more = True
        self.n_epochs = 0

        self.train_generator = DataGenerator(file = self.file, metadata = self.metadata,
                patients = self.patients_train, batch_size = self.batch_size, n_pixels = self.n_pixels)
        self.validation_generator = DataGenerator(file = self.file, metadata = self.metadata,
                patients = self.patients_test, batch_size = self.batch_size, n_pixels = self.n_pixels)

        while train_more:
            self.n_epochs += 1

            if self.verbose:
                print("[TRAIN_HELPER] On %d-th epoch of training model" % self.n_epochs)

            results = self.model.fit(
                       self.train_generator,
                       callbacks = callbacks_list,
                       #workers=12, use_multiprocessing=True,
                       #max_queue_size = 100,
                       validation_data = self.validation_generator)

            prediction = self.model.predict([self.X_test], batch_size = 128)
            self.summary["predictions"].append(prediction)
    
            # TODO: evaluate all metrics with prediction and append to summary

            val_loss = results.history['val_loss'][0]
            train_loss = results.history['loss'][0]

            self.summary["metrics"]["binary_crossentropy"].append(val_loss)
            self.summary["metrics_train"]["binary_crossentropy"].append(train_loss)

            if (val_loss * (1. + self.delta)) < self.best_loss:
                print("[TRAIN_HELPER] Loss improved by %.2f percent (%.3f -> %.3f), continuing for another epoch" % ( ((self.best_loss - val_loss) / val_loss) * 100., self.best_loss, val_loss) )
                self.best_loss = val_loss
                self.bad_epochs = 0

            else:
                print("[TRAIN_HELPER] Change in loss was %.2f percent (%.3f -> %.3f), incrementing bad epochs by 1." % ( ((self.best_loss - val_loss) / val_loss) * 100., self.best_loss, val_loss) ) 
                self.bad_epochs += 1

            if (self.increase_batch or self.decay_learning_rate) and self.bad_epochs >= 1: # increase batch size (decay learning rate as well?)
                if self.batch_size * 4 <= self.max_batch:
                    print("[TRAIN_HELPER] Increasing batch size from %d -> %d, resetting bad epochs to 0, and continuing for another epoch." % (self.batch_size, self.batch_size * 4))
                    self.batch_size *= 4
                    self.bad_epochs = 0

            if self.bad_epochs >= self.early_stopping_rounds:
                print("[TRAIN_HELPER] Number of early stopping rounds (%d) without improvement in loss of at least %.2f percent exceeded. Stopping training after %d epochs." % (self.early_stopping_rounds, self.delta*100., self.n_epochs))
                train_more = False
        
            if self.max_epochs > 0 and self.n_epochs >= self.max_epochs:
                print("[TRAIN_HELPER] Maximum number of training epochs (%d) reached. Stopping training." % (self.max_epochs))
                train_more = False

    def get_patients(self):
        self.patients = [pt for pt in self.metadata.keys() if "patient" in pt]
        self.data_manager = {}
        for pt in self.patients:
            self.data_manager[pt] = []
            for entry in self.metadata[pt]:
                self.data_manager[pt].append( { "keys" : [entry["X"], entry["y"]], "n_pneumonia" : float(entry["n_pneumonia"]) } )

        if self.n_pixels == -1:
            X = numpy.array(self.file[self.patients[0] + "_X_0"])
            self.n_pixels = X.shape[1]
            self.unet_config["n_pixels"] = self.n_pixels

    def load_from_file(self, patient):
        f_in = h5py.File(self.file_dict[patient]["file"], "r")
        X = numpy.array(f_in[self.file_dict[patient]["X"]])
        y = numpy.array(f_in[self.file_dict[patient]["y"]])

        return X, y

    def load_features(self, patients):
        X = []
        y = []

        for patient in patients:
            if len(X) == 0:
                X, y = self.load_from_file(patient)
            else:
                X_, y_ = self.load_from_file(patient)
                X = numpy.concatenate([X, X_])
                y = numpy.concatenate([y, y_])

        X = X.reshape([-1, self.n_pixels, self.n_pixels, 1])
        y = y.reshape([-1, self.n_pixels, self.n_pixels, 1])

        return X, y

    def initialize_model(self):
        self.model = models.unet(self.unet_config)

    def assess(self): # make plots of original | truth | pred \\ original + truth | original + pred | original + (pred - truth)
        random_idx = numpy.random.randint(0, self.n_instance_test, self.n_assess) 
        for idx, rand_idx in zip(range(self.n_assess), random_idx):
            image = self.X_test[rand_idx].reshape([self.n_pixels, self.n_pixels])
            truth = self.y_test[rand_idx].reshape([self.n_pixels, self.n_pixels])
            pred  = self.summary["predictions"][-1][rand_idx].reshape([self.n_pixels, self.n_pixels])

            utils.plot_image_truth_and_pred(image, truth, pred, "comp_%d" % idx)

