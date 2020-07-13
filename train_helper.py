import os, sys
import argparse
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
    def __init__(self, file, metadata, input_shape, patients, batch_size, 
                 verbose=False):
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

        return numpy.array(X), numpy.array(y)

    def get_random_slice(self):
        """
        Produces a random input and label np array consisting of a 
        random CT slice of interest, as well as n slices above and 
        n slices below.

        Returns an input array of shape (M,M,2n+1) where M is the 
        number of pixels in a slice.
        """
        M, M_, n_ = self.input_shape # assuming (M,M,2*n+1)
        n = int((n_-1.0)/2.0)
        f = self.file
        # Get random patient and index of slice of interest
        patient = random.choice(self.patients)
        n_slices = len(self.metadata[patient])
        slice_idx = random.randrange(0, n_slices)
        # Label
        y_stack = [f[patient+"_y_"+str(slice_idx)]]
        # Stack n input slices below and n above slice of interest
        X_stack = []
        for idx in range(slice_idx-n, slice_idx+n+1):
            if idx < 0 or idx > n_slices-1:
                # Set slices outside of boundary to zero
                X_stack.append(numpy.zeros((M, M)))
            elif idx == slice_idx:
                X_stack.append(f[patient+"_X_"+str(slice_idx)])
            else:
                X_stack.append(f[patient+"_X_"+str(idx)])

        return numpy.dstack(X_stack), numpy.dstack(y_stack)

class TrainHelper():
    def __init__(self):
        # CLI
        cli = argparse.ArgumentParser()
        # General
        cli.add_argument("-v", "--verbose", action="store_true", default=True)
        cli.add_argument("--fast", action="store_true", default=False)
        cli.add_argument(
            "--tag", 
            help="tag to identify this set", 
            type=str, 
            default=""
        )
        cli.add_argument(
            "--random_seed",
            help="random seed for test/train split",
            type=int,
            default=0
        )
        # Data
        cli.add_argument(
            "--input", 
            help="hdf5 file with data", 
            type=str
        )
        cli.add_argument(
            "--input_metadata", 
            help="json file with metadata", 
            type=str
        )
        cli.add_argument(
            "--summary_json",
            help="json file with summary of already trained model",
            type=str,
            default=""
        )
        cli.add_argument(
            "--n_extra_slices", 
            help="extra slices above and below input", 
            type=int, 
            default=0
        )
        # Hyperparameters
        cli.add_argument(
            "--max_epochs", 
            help="maximum number of training epochs", 
            type=int, 
            default=20
        )
        cli.add_argument(
            "--training_batch_size", 
            help="batch size for training", 
            type=int, 
            default=16
        )
        cli.add_argument(
            "--validation_batch_size", 
            help="batch size for validation", 
            type=int, 
            default=16
        )
        cli.add_argument(
            "--max_batch_size", 
            help="maximum batch size", 
            type=int, 
            default=16
        )
        cli.add_argument(
            "--train_frac",
            help="fraction of input used for training",
            type=float,
            default=0.7
        )
        cli.add_argument(
            "--delta",
            help="Percent by which loss must improve",
            type=float,
            default=0.01
        )
        cli.add_argument(
            "--early_stopping_rounds",
            help="Percent by which loss must improve",
            type=int,
            default=2
        )
        cli.add_argument(
            "--increase_batch",
            help="Increase batch if more than one 'bad' epoch",
            action="store_true",
            default=True
        )
        cli.add_argument(
            "--decay_learning_rate",
            help="Decay learning rate if more than one 'bad' epoch",
            action="store_true",
            default=False
        )
        # Load CLI args into namespace
        cli.parse_args(namespace=self)
        # Load data
        self.input_shape = (-1, -1, -1) # i.e. (M,M,2n+1)
        self.data_manager = {}
        self.n_pixels = -1 # i.e. M
        self.load_data() # sets the above three variables
        # Set loss tracker
        self.best_loss = 999999.0
        # Initialize results object
        self.summary = {
            "input": self.input,
            "train_frac": self.train_frac,
            "config": {},
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
        # Initialize data generators
        self.training_generator = DataGenerator(
            file=self.file,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_train,
            batch_size=self.training_batch_size
        )
        self.validation_generator = DataGenerator(
            file=self.file,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=self.validation_batch_size
        )

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
        # Derive/store input shape
        X_ = numpy.array(self.file[self.patients[0]+"_X_0"])
        self.n_pixels = X_.shape[0]
        self.input_shape = (X_.shape[0], X_.shape[1], 
                            2*self.n_extra_slices+1)

        self.calculate_pneumonia_imbalance()
        # Calculate number of training/testing slices
        self.n_train = int(self.train_frac*float(len(self.patients)))
        self.n_test  = len(self.patients)-self.n_train
        # Shuffle patients, fixing random seed for reproducibility
        # Note: for more rigorous comparisons we should do k-fold validation 
        # with multiple different test/train splits
        patients_shuffle = self.patients
        random.seed(self.random_seed)
        random.shuffle(patients_shuffle)
        # Distribute training and testing data
        self.patients_train = patients_shuffle[:self.n_train]
        self.patients_test  = patients_shuffle[self.n_train:]

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
        return

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
        return

    def load_weights(self, model, model_config, weights):
        self.summary["config"] = model_config
        self.model = model
        self.model.load_weights(weights)
        return

    def train(self, model, model_config=None):
        self.summary["config"] = model_config
        self.model = model
        self.train_with_early_stopping()
        return

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
            self.n_epochs += 1

            if self.verbose:
                print("[TRAIN_HELPER] On epoch %d of training model" 
                      % self.n_epochs)

            results = self.model.fit(
                self.training_generator,
                callbacks=callbacks_list,
                use_multiprocessing=False,
                validation_data=self.validation_generator
            )

            #prediction = self.model.predict(self.validation_generator)
            #self.summary["predictions"].append(prediction)
    
            # TODO: evaluate all metrics with prediction and append to summary

            val_loss = results.history["val_loss"][0]

            for metric in ["loss", "accuracy", "dice_loss"]:
                self.summary["metrics"][metric].append(
                    str(results.history["val_" + metric][0])
                )
                self.summary["metrics_train"][metric].append(
                    str(results.history[metric][0])
                )

            percent_change = ((self.best_loss - val_loss)/val_loss)*100.0
            if (val_loss*(1. + self.delta)) < self.best_loss:
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

            if ((self.increase_batch or self.decay_learning_rate) 
                and self.bad_epochs >= 1): 
                # Increase batch size (decay learning rate as well?)
                if self.training_batch_size*4 <= self.max_batch_size:
                    print("[TRAIN_HELPER] --> Increasing batch size from %d -> %d" 
                          % (self.training_batch_size, self.training_batch_size*4))
                    print("[TRAIN_HELPER] --> resetting bad epochs to 0")
                    print("[TRAIN_HELPER] --> continuing for another epoch")
                    self.training_batch_size *= 4
                    self.training_generator.batch_size = self.training_batch_size
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

        return

    def write_metadata(self):
        """
        Write json summarizing training metadata
        """

        self.summary["weights"] = self.weights_file.replace("{epoch:02d}", "01")

        with open("results_%s.json" % self.tag, "w") as f_out:
            json.dump(self.summary, f_out, indent = 4, sort_keys = True)

        return

    def make_roc_curve(self):
        """
        Calculate tpr, fpr, auc and uncertainties with n_jackknife jackknife 
        samples each of size n_batches * validation generator batch size
        """

        n_jackknife = 2
        n_batches = 3

        self.tprs = []
        self.fprs = []
        self.aucs = []

        # Note: looping through the jackknife samples is not a very good way
        # to calculate the uncertainty. What should be done instead is
        # calculating mean and std dev over multiple trainings each with
        # different test/train splits. This is done for example in 
        # scripts/compare_2d_vs_2point5d.py
        # Keeping the functionality below to prevent breaking the plotting
        # scripts down the line. Should be changed soon (lazy, I know).
        for i in range(n_jackknife): # number of bootstrap samples
            pred = []
            y = []
            for j in range(n_batches): # number of validation set batches
                X, y_ = self.validation_generator.__getitem__((i*n_batches)+j)
                pred_ = self.model.predict(X)
                y.append(y_)
                if j == 0:
                    pred = pred_
                else:
                    pred = numpy.concatenate([pred, pred_])

            pred = numpy.array(pred)
            y = numpy.array(y)

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

        roc_metrics = [
            tpr_mean.tolist(),
            tpr_std.tolist(),
            fpr_mean.tolist(),
            fpr_std.tolist(),
            auc,
            auc_std
        ]
        roc_metric_labels = [
            "tpr_mean",
            "tpr_std",
            "fpr_mean",
            "fpr_std",
            "auc",
            "auc_std"
        ]

        for metric, label in zip(roc_metrics, roc_metric_labels):
            self.summary[label] = metric

        utils.plot_roc(fpr_mean, fpr_std, tpr_mean, tpr_std, auc, auc_std, self.tag)
        return

    def assess(self, n_plots=5): 
        """
        Make plots of (orig|truth|pred)\\(orig+truth|orig+pred|original+(pred-truth))
        """

        plot_summary = []

        slice_index = self.n_extra_slices # index of slice of interest
        for i in range(n_plots):
            input_data, truth = self.validation_generator.get_random_slice()
            pred = self.model.predict(numpy.array([input_data]), batch_size=1)

            # Add image with corresponding dice coefficient to a dict
            dice = str(
                    numpy.array(
                        metrics.dice_loss(truth, pred)
                    ).flatten()[0])
            plot_summary.append([dice, "comp_%d" % i]) 

            # skip images with no pneumonia
            if dice == 1:
                i -= 1
                continue

            # Extract slice of interest from input data
            orig = input_data[:,:,slice_index]
            # Reshape into plottable images
            orig = orig.reshape([self.n_pixels, self.n_pixels])
            truth = truth.reshape([self.n_pixels, self.n_pixels])
            pred = pred.reshape([self.n_pixels, self.n_pixels])       
            # Plot
            utils.plot_image_truth_and_pred(
                orig, 
                truth, 
                pred, 
                self.tag+"_comp_%d" % i
            )

        self.summary["qualitative_assessments"] = plot_summary

        return
