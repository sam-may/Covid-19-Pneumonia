import re
import h5py
import json
import random
import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from helpers.train_helper import TrainHelper
from helpers.print_helper import print
from models.cnn import cnn3D as cnn
from generators.cnn import DataGenerator3D
from plots.utils import calc_auc

class CNNHelper(TrainHelper):
    def __init__(self):
        super().__init__()
        self.cli.add_argument(
            "--delta",
            help="Percent by which loss must improve",
            type=float,
            default=0.01
        )
        self.cli.add_argument(
            "--no_early_stopping",
            help="Disable early stopping",
            action="store_true",
            default=False
        )
        self.cli.add_argument(
            "--early_stopping_rounds",
            help="Percent by which loss must improve",
            type=int,
            default=2
        )
        self.cli.add_argument(
            "--increase_batch",
            help="Increase batch if more than one 'bad' epoch",
            action="store_true",
            default=False
        )
        self.cli.add_argument(
            "--decay_learning_rate",
            help="Decay learning rate if more than one 'bad' epoch",
            action="store_true",
            default=False
        )
        self.cli.add_argument(
            "--dice_smooth",
            help="Smoothing factor to put in num/denom of dice coeff",
            type=float,
            default=1
        )
        self.cli.add_argument(
            "--bce_alpha",
            help="Weight for positive instances in binary x-entropy",
            type=float,
            default=3
        )
        self.cli.add_argument(
            "--loss_function",
            help="Loss function to use for training (from models/loss_functions.py)",
            type=str,
            default="weighted_crossentropy"
        ) 
        self.cli.add_argument(
            "--roc_batches",
            help="Number of batches for calculating ROC metrics",
            type=int,
            default=3
        ) 
        self.cli.add_argument(
            "--extra_features",
            help="Space-separated list of extra features to pass to model",
            type=str,
            nargs="*"
        ) 
        self.parse_cli()

    def load_data(self):
        """
        Load input hdf5 file and metadata json, interpret and set attributes 
        listed below:

        self.data
        self.metadata
        self.patients
        self.input_shape
        """
        self.data = h5py.File(self.data_hdf5, "r") 
        with open(self.metadata_json, "r") as f_in:
            self.metadata = json.load(f_in)
        # Get list of patients 
        self.patients = list(self.data.keys())
        # Derive/store input shape
        self.input_shape = self.data[self.patients[0]].shape
        # Derive/store number of extra features
        self.n_extra_features = 0
        if self.extra_features:
            for feature_name in self.extra_features:
                feature = self.metadata[self.patients[0]][feature_name]
                if type(feature) == list:
                    self.n_extra_features += len(feature)
                else:
                    self.n_extra_features += 1
        return

    def train(self):
        """Train model with early stopping"""
        training_batch_size = self.training_batch_size
        validation_batch_size = self.validation_batch_size
        # Initialize data generators
        training_generator = DataGenerator3D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_train,
            batch_size=training_batch_size,
            input_reshape=(64, 64, 64),
            extra_features=self.extra_features,
            do_rotations=True
        )
        validation_generator = DataGenerator3D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=validation_batch_size,
            input_reshape=(64, 64, 64),
            extra_features=self.extra_features
        )
        # Write weights to hdf5 each epoch
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_file)
        callbacks_list = [checkpoint]
        # Training loop
        train_more = True
        epoch_num = 0
        bad_epochs = 0
        best_loss = 999999.0
        while train_more:
            epoch_num += 1
            if self.verbose:
                print("On epoch %d of training model" % epoch_num)
            # Run training
            results = self.model.fit(
                training_generator,
                callbacks=callbacks_list,
                use_multiprocessing=False,
                validation_data=validation_generator,
                epochs=epoch_num,
                initial_epoch=epoch_num-1
            )
            # Calculate TPR, FPR, and AUC
            y = []
            for i in range(self.roc_batches):
                X, y_ = validation_generator.__getitem__(0)
                pred_ = self.model.predict(X)
                y.append(y_)
                if i == 0:
                    pred = pred_
                else:
                    pred = numpy.concatenate([pred, pred_])
            pred = numpy.array(pred)
            y = numpy.array(y)
            fpr, tpr, auc = calc_auc(y.flatten(), pred.flatten())
            results.history["fpr"] = fpr
            results.history["tpr"] = tpr
            results.history["auc"] = auc
            # Update epoch metrics
            print("Saving epoch metrics")
            self.save_metrics(results.history)
            # Stop training after epoch cap
            if self.max_epochs > 0 and epoch_num >= self.max_epochs:
                print("Maximum number of training epochs (%d) reached" 
                      % (self.max_epochs))
                print("--> stopped training")
                train_more = False
            if self.no_early_stopping:
                continue
            # Calculate % change for early stopping
            val_loss = results.history["val_loss"][0]
            percent_change = ((best_loss - val_loss)/val_loss)*100.0
            if (val_loss*(1. + self.delta)) < best_loss:
                print("Loss improved by %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, best_loss, val_loss))
                print("--> continuing for another epoch")
                best_loss = val_loss
                bad_epochs = 0
            else:
                print("Change in loss was %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, best_loss, val_loss)) 
                print("--> incrementing bad epochs by 1")
                bad_epochs += 1
            # Handle dynamic batch size and/or learning rate
            if ((self.increase_batch or self.decay_learning_rate) 
                and bad_epochs >= 1): 
                # Increase batch size (decay learning rate as well?)
                if training_batch_size*4 <= self.max_batch_size:
                    print("--> increasing batch size from %d -> %d" 
                          % (training_batch_size, training_batch_size*4))
                    print("--> resetting bad epochs to 0")
                    print("--> continuing for another epoch")
                    training_batch_size *= 4
                    training_generator.batch_size = training_batch_size
                    bad_epochs = 0
            # Check for early stopping
            if bad_epochs >= self.early_stopping_rounds:
                print("Exceeded patience (%d)" % self.early_stopping_rounds)
                print("--> stopping training after %d epochs" 
                      % (epoch_num))
                train_more = False

        self.make_plots()
        return

    def make_plots(self):
        print("Making assessment plots")
        training = self.cur_training
        plot_dir = self.out_dir+"plots/"
        # Loss timeseriese
        df = pandas.DataFrame(self.metrics)
        df = df[df.training == training].reset_index(drop=True)
        # Plot
        fig, axes = plt.subplots()
        plt.plot(df.loss, label="training")
        plt.plot(df.val_loss, label="validation")
        # Plot formatting
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Weighted Crossentropy")
        plt.title("Loss Timeseries")
        plt.savefig(plot_dir+"loss_timeseries_"+str(training)+".png")
        plt.close(fig)
        # Get false/true positive rates, AUC
        validation_generator = DataGenerator3D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=4,
            input_reshape=(64, 64, 64),
            extra_features=self.extra_features
        )
        labels = []
        preds = []
        for i in range(len(validation_generator)):
            # Get data, label, and prediction
            X, y = validation_generator.__getitem__(0)
            pred = self.model.predict(X)
            # Add to list
            labels.append(y)
            if i == 0:
                preds = pred
            else:
                preds = numpy.concatenate([preds, pred])
        labels = numpy.array(labels)
        preds = numpy.array(preds)
        # Get false/true positive rates, AUC
        fprs, tprs, auc = calc_auc(labels.flatten(), preds.flatten())
        # ROC Curve
        fig, axes = plt.subplots()
        axes.plot(fprs, tprs, label="%s [AUC: %.3f]" % (self.tag, auc))
        # Formatting
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(plot_dir+"roc_curve_"+str(training)+".png")
        plt.close(fig)

        return
        

if __name__ == "__main__":
    # Initialize helper
    cnn_helper = CNNHelper()
    # Initialize model
    cnn3D_config = {
        "input_shape": (64, 64, 64, 2),
        "n_extra_features": cnn_helper.n_extra_features,
        "dropout": 0.25,
        "batch_norm": False,
        "learning_rate": 0.0000005,
        "bce_alpha": cnn_helper.bce_alpha,
        "dice_smooth": cnn_helper.dice_smooth,
        "loss_function": cnn_helper.loss_function 
    }
    model = cnn(cnn3D_config)
    # Train model
    cnn_helper.run_training(model, cnn3D_config)
