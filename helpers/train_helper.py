import os
import argparse
import numpy
import h5py
import random
import json

def train_decorator(train_func):
    def decorated_func(*args):
        # Get helper class instance
        self = args[0]
        # Set up directories
        organized = self.organize()
        if not organized:
            return
        # Run training
        result = train_func(*args)
        # Save summary files
        self.summarize()

        return result

    return decorated_func


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
        cli.add_argument(
            "--loss_function",
            help="Loss function to use during training",
            type=str,
            default="weighted_crossentropy"
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
            "calc_dice_loss": [],
            "calc_dice_loss_train": [],
            "calc_weighted_crossentropy": [],
            "calc_weighted_crossentropy_train": []
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

    def organize(self):
        """Set up directory structure where model is to be saved"""
        print("[TRAIN_HELPER] Writing output files to "+self.out_dir)
        if not os.path.isdir("trained_models"):
            os.mkdir("trained_models")
        if os.path.isfile(self.summary_file):
            print("[TRAIN_HELPER] Model with this tag already trained")
            print("[TRAIN_HELPER] --> check or delete "+self.out_dir)
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
