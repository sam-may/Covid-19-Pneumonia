import os
import argparse
import numpy
import h5py
import random
import json
import tensorflow.keras as keras
import loss_functions
import utils
from timeit import default_timer as timer

class DataGenerator(keras.utils.Sequence):
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

class TrainHelper():
    """
    An object to wrap the training process with useful and necessary functions
    """
    def __init__(self):
        # Command Line Interface (CLI)
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
            "--data_hdf5", 
            help="hdf5 file with data", 
            type=str
        )
        cli.add_argument(
            "--metadata_json", 
            help="json file with metadata", 
            type=str
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
        cli.add_argument(
            "--dice_smooth",
            help="Smoothing factor to put in num/denom of dice coeff",
            type=float,
            default=1
        )
        cli.add_argument(
            "--bce_alpha",
            help="Weight for positive instances in binary x-entropy",
            type=float,
            default=3
        )
        # Load CLI args into namespace
        cli.parse_args(namespace=self)
        # Load/calculate various training parameters
        self.load_data()
        # Set loss tracker
        self.best_loss = 999999.0
        # Initialize directory to hold output files
        self.out_dir = "trained_models/"+self.tag+"/"
        # Initialize weights file, updated each epoch
        self.weights_file = (self.out_dir
                             + "weights/"
                             + self.tag
                             + "_weights_{epoch:02d}.hdf5")
        # Initialize metrics, these here are updated each epoch
        self.metrics = {
            "loss": [],
            "loss_train": [],
            "accuracy": [],
            "accuracy_train": [],
            "dice_loss": [],
            "dice_loss_train": []
        }
        self.metrics_file = self.out_dir+self.tag+"_metrics.npz"
        # Initialize results object, written at end of training
        self.summary = {
            "train_params": vars(cli.parse_args()),
            "model_config": {}, # set by self.train
            "weights": self.weights_file.replace("{epoch:02d}", "01"),
            "metrics": self.metrics_file,
            "patients_test": self.patients_test
        }
        self.summary["train_params"]["input_shape"] = self.input_shape
        self.summary_file = self.out_dir+self.tag+"_summary.json"
        # Initialize data generators
        self.training_generator = DataGenerator(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_train,
            batch_size=self.training_batch_size
        )
        self.validation_generator = DataGenerator(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=self.validation_batch_size
        )

    def load_data(self):
        """
        Load input hdf5 file and metadata json, interpret and set attributes 
        listed below:

        self.data
        self.metadata
        self.patients
        self.input_shape
        self.pneumonia_fraction
        self.patients_train
        self.patients_test
        """
        self.data = h5py.File(self.data_hdf5, "r") 
        with open(self.metadata_json, "r") as f_in:
            self.metadata = json.load(f_in)
        # Get list of patients 
        self.patients = [k for k in self.metadata.keys() if "patient" in k]
        # Derive/store input shape
        X_ = numpy.array(self.data[self.patients[0]+"_X_0"])
        n_pixels = X_.shape[0]
        self.input_shape = (X_.shape[0], X_.shape[1], 2*self.n_extra_slices+1)
        # Calculate pneumonia imbalance
        pneumonia_pixels = 0
        all_pixels = 0
        for patient in self.patients:
            for entry in self.metadata[patient]:
                pneumonia_pixels += float(entry["n_pneumonia"])
                all_pixels += float(n_pixels**2)
        self.pneumonia_fraction = pneumonia_pixels/all_pixels
        print("[TRAIN_HELPER] Fraction of pixels with pneumonia: %.6f" 
              % self.pneumonia_fraction)
        # Calculate number of training/testing slices
        n_train = int(self.train_frac*float(len(self.patients)))
        # Shuffle patients, fixing random seed for reproducibility
        # Note: for more rigorous comparisons we should do k-fold validation 
        # with multiple different test/train splits
        patients_shuffle = self.patients
        random.seed(self.random_seed)
        random.shuffle(patients_shuffle)
        # Distribute training and testing data
        self.patients_train = patients_shuffle[:n_train]
        self.patients_test = patients_shuffle[n_train:]
        return

    def train(self, model, model_config):
        """Train model with early stopping"""
        # Set up directories
        organized = self.organize()
        if not organized:
            return
        # Store model config and model
        self.summary["model_config"] = model_config
        self.model = model
        # Write weights to hdf5 each epoch
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
            # Run training
            results = self.model.fit(
                self.training_generator,
                callbacks=callbacks_list,
                use_multiprocessing=False,
                validation_data=self.validation_generator
            )
            # Update epoch metrics
            print("[TRAIN_HELPER] Saving epoch metrics")
            for name in ["loss", "accuracy", "dice_loss"]:
                self.metrics[name+"_train"].append(results.history[name][0])
                self.metrics[name].append(results.history["val_"+name][0])
            # Calculate % change for early stopping
            val_loss = results.history["val_loss"][0]
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
            # Handle dynamic batch size and/or learning rate
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
            # Check for early stopping
            if self.bad_epochs >= self.early_stopping_rounds:
                print("[TRAIN_HELPER] Number of early stopping rounds (%d) without\
                      improvement in loss of at least %.2f percent exceeded" 
                      % (self.early_stopping_rounds, self.delta*100.))
                print("[TRAIN_HELPER] --> stopping training after %d epochs" 
                      % (self.n_epochs))
                train_more = False
            # Stop training after epoch cap
            if self.max_epochs > 0 and self.n_epochs >= self.max_epochs:
                print("[TRAIN_HELPER] Maximum number of training epochs (%d) reached" 
                      % (self.max_epochs))
                print("[TRAIN_HELPER] --> stopped training")
                train_more = False
        # Save summary info
        self.summarize()
        return

    def organize(self):
        """Set up directory structure where model is to be saved"""
        print("[TRAIN_HELPER] Writing output files to "+self.out_dir)
        if not os.path.isdir("trained_models"):
            os.mkdir("trained_models")
        if os.path.isfile(self.summary_file):
            print("[TRAIN_HELPER] Model with this tag already trained")
            print("[TRAIN_HELPER] --> check/delete "+self.out_dir)
            print("[TRAIN_HELPER] --> aborting training")
            return False
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        if not os.path.isdir(self.out_dir+"weights"):
            os.mkdir(self.out_dir+"weights")
        if not os.path.isdir(self.out_dir+"plots"):
            os.mkdir(self.out_dir+"plots")

        return True

    def summarize(self):
        """
        Calculate additional performance metrics of final model and write 
        summary and metrics files
        """
        print("[TRAIN_HELPER] Saving summary and additional metrics")
        # Write summary to json
        with open(self.summary_file, "w") as f_out:
            json.dump(self.summary, f_out, indent=4, sort_keys=True)
        # Convert all metrics to numpy arrays
        for name, metric in self.metrics.items():
            self.metrics[name] = numpy.array(metric)
        # Write metrics to compressed npz
        numpy.savez_compressed(self.metrics_file, **self.metrics)
        return

if __name__ == "__main__":
    import models
    # Initialize helper
    helper = TrainHelper()
    # Initialize model
    unet_config = {
        "input_shape": helper.input_shape,
        "n_filters": 12,
        "n_layers_conv": 2,
        "n_layers_unet": 3,
        "kernel_size": (4, 4),
        "dropout": 0.0,
        "batch_norm": False,
        "learning_rate": 0.00005,
        "alpha": helper.bce_alpha,
        "dice_smooth": helper.dice_smooth
    }
    model = models.unet(unet_config)
    # Train model
    helper.train(model, unet_config)
