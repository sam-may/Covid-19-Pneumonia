import os, sys

import numpy
import h5py
import random

import models
import utils

import tensorflow
import tensorflow.keras as keras

class Train_Helper():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.fast           = kwargs.get('fast', False)

        self.input          = kwargs.get('input')
        self.tag            = kwargs.get('tag')
        self.verbose        = kwargs.get('verbose', True)

        self.train_frac     = kwargs.get('train_frac', 0.5)

        self.unet_config    = kwargs.get('unet_config', {
                                            "n_filters" : 24,
                                            "n_layers_conv" : 1,
                                            "n_layers_unet" : 4,
                                            "kernel_size" : 3,
                                            "dropout" : 0.1,
                                            "batch_norm" : True,
                                            "learning_rate" : 0.001,
                                        })

        self.best_loss = 999999
        self.delta = 0.01 # percent by which loss must improve to be considered an improvement
        self.early_stopping_rounds = 3
    
        self.increase_batch = False
        self.batch_size = 16
        self.max_epochs = 20

        self.n_assess = 10

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
        if self.file is not None:
            print("[TRAIN_HELPER] Successfully opened file %s" % self.input)
        else:
            print("[TRAIN_HELPER] Error opening file %s" % self.input)
            sys.exit(1)

        self.get_patients()

        self.n_train = int(self.train_frac * float(len(self.patients)))
        self.n_test  = len(self.patients) - self.n_train

        patients_shuffle = self.patients
        random.shuffle(patients_shuffle)

        self.patients_train = patients_shuffle[:self.n_train]
        self.patients_test  = patients_shuffle[self.n_train:]

        if self.fast:
            self.patients_train = [self.patients_train[0]]
            self.patients_test = [self.patients_test[0]]
            

        self.X_train, self.y_train = self.load_features(self.patients_train)
        self.X_test, self.y_test = self.load_features(self.patients_test)

        self.n_pixels = self.X_train.shape[1]
        self.unet_config["n_pixels"] = self.n_pixels

        self.n_instance_train = self.X_train.shape[0]
        self.n_instance_test = self.X_test.shape[0]

        if self.verbose:
            print("[TRAIN_HELPER] Images and labels have dimensions %d x %d" % (self.n_pixels, self.n_pixels))
            print("[TRAIN_HELPER] Training with %d instances" % self.n_instance_train)
            print("[TRAIN_HELPER] Testing with %d instances" % self.n_instance_test)

    def load_weights(self, weights):
        if self.model is not None:
            self.initialize_model()
        self.model.load_weights(weights)

    def train(self):
        self.initialize_model()
        self.train_with_early_stopping()

    def train_with_early_stopping(self):
        self.weights_file = "weights/" + self.tag + "_weights_{epoch:02d}.hdf5"
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_file) # save after every epoch
        callbacks_list = [checkpoint]

        train_more = True
        self.n_epochs = 0

        while train_more:
            self.n_epochs += 1

            if self.verbose:
                print("[TRAIN_HELPER] On %d-th epoch of training model" % self.n_epochs)
            results = self.model.fit([self.X_train], self.y_train,
                       epochs = 1, batch_size = self.batch_size,
                       callbacks = callbacks_list,
                       validation_data = (self.X_test, self.y_test))

            prediction = self.model.predict([self.X_test], batch_size = 1024)
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

            if self.increase_batch and self.bad_epochs >= 1: # increase batch size (decay learning rate as well?)
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
        keys = self.file.keys()
        self.patients = [key.replace("_X", "") for key in keys if "_X" in key]
        print("[TRAIN_HELPER] Found %d patients features and labels" % len(self.patients))

    def load_features(self, patients):
        X = []
        y = []

        self.n_pixels = numpy.array(self.file[patients[0] + "_X"]).shape[1]

        for patient in patients:
            if len(X) == 0:
                X = numpy.array(self.file[patient + "_X"])
                y = numpy.array(self.file[patient + "_y"])
            else:
                X = numpy.concatenate([X,  numpy.array(self.file[patient + "_X"])])
                y = numpy.concatenate([y,  numpy.array(self.file[patient + "_y"])])

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

        


