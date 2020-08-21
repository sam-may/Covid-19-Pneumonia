import os
import argparse
import numpy
import pandas
import random
import json
from .print_helper import print

class TrainHelper():
    """
    An object to wrap the training process with useful and necessary functions
    """
    def __init__(self):
        # Command Line Interface (CLI)
        self.cli = argparse.ArgumentParser()
        # General
        self.cli.add_argument("-v", "--verbose", action="store_true", default=True)
        self.cli.add_argument("--fast", action="store_true", default=False)
        self.cli.add_argument(
            "--tag", 
            help="tag to identify this set", 
            type=str, 
            default=""
        )
        self.cli.add_argument(
            "--random_seed",
            help="random seed for test/train split",
            type=int,
            default=0
        )
        # Data
        self.cli.add_argument(
            "--data_hdf5", 
            help="hdf5 file with data", 
            type=str
        )
        self.cli.add_argument(
            "--metadata_json", 
            help="json file with metadata", 
            type=str
        )
        self.cli.add_argument(
            "--n_trainings", 
            help="Number of trainings with random test/train splits", 
            type=int, 
            default=1
        )
        # Hyperparameters
        self.cli.add_argument(
            "--max_epochs", 
            help="maximum number of training epochs", 
            type=int, 
            default=20
        )
        self.cli.add_argument(
            "--train_frac",
            help="Fraction of data to use for training",
            type=float,
            default=0.7
        )
        self.cli.add_argument(
            "--training_batch_size", 
            help="batch size for training", 
            type=int, 
            default=16
        )
        self.cli.add_argument(
            "--validation_batch_size", 
            help="batch size for validation", 
            type=int, 
            default=16
        )
        self.cli.add_argument(
            "--max_batch_size", 
            help="maximum batch size", 
            type=int, 
            default=16
        )

    def parse_cli(self):
        # Load CLI args into namespace
        self.cli.parse_args(namespace=self)
        # Load/calculate various training parameters
        self.load_data()
        # Set trackers
        self.best_loss = 999999.0
        # Initialize directory to hold output files
        self.out_dir = "trained_models/"+self.tag+"/"
        # Initialize weights file, updated each epoch
        self.weights_file = (self.out_dir
                             + "weights/"
                             + self.tag
                             + "_weights_{epoch:02d}-{val_loss:.2f}.hdf5")
        # Initialize metrics
        self.metrics = []
        self.metrics_file = self.out_dir+self.tag+"_metrics.pickle"
        # Initialize results object, written at end of training
        self.summary = {
            "train_params": vars(self.cli.parse_args()),
            "model_config": {}, # set by self.train
            "patients_test": [],
            "random_seeds": []
        }
        self.summary["train_params"]["input_shape"] = self.input_shape
        self.summary_file = self.out_dir+self.tag+"_summary.json"
        return

    def load_data(self):
        """
        Load all relevant data. MUST set the following attributes:

        self.data
        self.patients
        self.input_shape

        MUST BE OVERWRITTEN
        """
        raise NotImplementedError

    def shuffle_patients(self, random_seed=0):
        # Calculate number of training/testing slices
        n_train = int(self.train_frac*float(len(self.patients)))
        # Shuffle patients, fixing random seed for reproducibility
        # Note: for more rigorous comparisons we should do k-fold validation 
        # with multiple different test/train splits
        patients_shuffle = self.patients
        random.seed(random_seed)
        random.shuffle(patients_shuffle)
        # Distribute training and testing data
        self.patients_train = patients_shuffle[:n_train]
        self.patients_test = patients_shuffle[n_train:]
        return

    def train(self):
        """MUST BE OVERWRITTEN"""
        raise NotImplementedError

    def run_training(self, model, model_config):
        """Run N trainings with different patient shuffles"""
        # Store model config and model
        self.summary["model_config"] = model_config
        self.model = model
        # Set up directories
        organized = self.organize()
        if not organized:
            return
        # Run training
        if self.n_trainings > 1:
            # One process running several trainings
            for i in range(self.n_trainings):
                self.random_seed = i
                self.shuffle_patients(random_seed=i)
                self.summary["patients_test"].append(self.patients_test)
                self.summary["random_seeds"].append(self.random_seed)
                self.train()
        else:
            # One process running a single training
            self.shuffle_patients(random_seed=self.random_seed)
            self.summary["patients_test"].append(self.patients_test)
            self.summary["random_seeds"].append(self.random_seed)
            self.train()
        # Wrap up
        self.summarize()
        return

    def save_metrics(self, epoch_metrics):
        if type(epoch_metrics) != dict:
            raise TypeError
        df_row = {"random_seed": self.random_seed}
        for key, value in epoch_metrics.items():
            if type(value) == list and len(value) == 1:
                df_row[key] = value[0]
            else:
                df_row[key] = value
        self.metrics.append(df_row)
        return

    def organize(self):
        """Set up directory structure where model is to be saved"""
        print("Writing output files to "+self.out_dir)
        if not os.path.isdir("trained_models"):
            os.mkdir("trained_models")
        if os.path.isfile(self.summary_file):
            print("Model with this tag already trained")
            print("--> check or delete "+self.out_dir)
            print("--> aborting training")
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
        print("Saving summary and additional metrics")
        # Write summary to json
        with open(self.summary_file, "w") as f_out:
            json.dump(self.summary, f_out, indent=4, sort_keys=True)
        # Write metrics dataframe to pickle file
        pandas.DataFrame(self.metrics).to_pickle(self.metrics_file)
        return
