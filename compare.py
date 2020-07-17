import os
import argparse
import json
import glob
import numpy
import h5py
import tensorflow
import tensorflow.keras as keras
from datetime import datetime as dt
import matplotlib.pyplot as plt
import loss_functions
import plots
import models
from train import DataGenerator

class ModelHelper():
    def __init__(self, model, model_dir):
        self.plot_dir = model_dir+"plots/"
        summary_json = glob.glob(model_dir+"*summary.json")[0]
        with open(summary_json, "r") as f_in:
            summary = json.load(f_in)
            # Load model
            self.model = model(summary["model_config"], verbose=False)
            train_params = summary["train_params"]
            self.patients_test = summary["patients_test"]
            # Training parameters
            self.tag = train_params["tag"]
            self.data_hdf5 = train_params["data_hdf5"]
            self.metadata_json = train_params["metadata_json"]
            self.input_shape = train_params["input_shape"]
            self.n_extra_slices = int((self.input_shape[-1] - 1)/2.0)
            self.validation_batch_size = train_params["validation_batch_size"]
            # External files
            self.metrics = dict(numpy.load(summary["metrics"]))
            self.model.load_weights(summary["weights"])

        # Plot-specific attributes
        self.dice_scores = []

    def assign_data(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.data_generator = DataGenerator(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=self.validation_batch_size
        )

    def plot_micro_dice(self, fig=None):
        """Histograms of the scores for 70% of validation cases"""
        # Get case-by-case dice scores
        dice_scores = self.dice_scores
        if not dice_scores:
            print("[MODEL_HELPER] Generating dice scores")
            for patient in self.patients_test:
                n_slices = len(self.metadata[patient])
                for i in range(int(0.25*n_slices)):
                    X, y = self.data_generator.get_random_slice(patient=patient)
                    y_pred = self.model.predict(numpy.array([X]), batch_size=1)
                    dice_scores.append(loss_functions.dice_loss(y, y_pred))
        # Plot
        plots.hist_1D(
            dice_scores, 
            numpy.linspace(0,1,21),
            self.plot_dir+self.tag+"_micro_dice.pdf",
            title="Micro Dice",
            xlabel="Dice Score",
            fig=fig,
            save=(not fig),
            tag=self.tag
        )

        if not fig:
            self.side_by_side_plots()

        return

    def plot_roc_curve(self, fig=None):
        """
        Calculate tpr, fpr, auc and uncertainties with n_jackknife jackknife 
        samples each of size n_batches*validation generator batch size
        """

        n_jackknife = 2
        n_batches = 3
        self.tprs = []
        self.fprs = []
        self.aucs = []
        for i in range(n_jackknife): # number of bootstrap samples
            pred = []
            y = []
            for j in range(n_batches): # number of validation set batches
                X, y_ = self.data_generator.__getitem__((i*n_batches)+j)
                pred_ = self.model.predict(X)
                y.append(y_)
                if j == 0:
                    pred = pred_
                else:
                    pred = numpy.concatenate([pred, pred_])

            pred = numpy.array(pred)
            y = numpy.array(y)
            # Get false/true positive rates, AUC
            fpr, tpr, auc = plots.calc_auc(y.flatten(), pred.flatten())
            self.fprs.append(fpr)
            self.tprs.append(tpr)
            self.aucs.append(auc)

        tpr_mean = numpy.mean(self.tprs, axis=0)
        tpr_std = numpy.std(self.tprs, axis=0)
        fpr_mean = numpy.mean(self.fprs, axis=0)
        fpr_std = numpy.std(self.fprs, axis=0)
        auc = numpy.mean(self.aucs)
        auc_std = numpy.std(self.aucs)
        plots.roc_plot(
            fpr_mean, 
            fpr_std, 
            tpr_mean, 
            tpr_std, 
            auc, 
            auc_std, 
            self.plot_dir+self.tag+"_roc_curve.pdf",
            fig=fig,
            save=(not fig),
            tag=self.tag
        )
        return

    def side_by_side_plots(self, n_plots=5, fig=None): 
        """
        Make plots of (orig|truth|pred)\\(orig+truth|orig+pred|original+(pred-truth))
        """
        slice_index = self.n_extra_slices # index of slice of interest
        for i in range(n_plots):
            input_data, truth = self.data_generator.get_random_slice()
            pred = self.model.predict(numpy.array([input_data]), batch_size=1)

            # Add image with corresponding dice coefficient to a dict
            dice_loss = numpy.array(loss_functions.dice_loss(truth, pred))
            dice = str(dice_loss.flatten()[0])

            # Skip images with no pneumonia
            if dice == 1:
                i -= 1
                continue

            # Extract slice of interest from input data
            orig = input_data[:,:,slice_index]
            # Reshape into plottable images
            M = self.input_shape[0]
            image = orig.reshape([M, M])
            truth = truth.reshape([M, M])
            pred = pred.reshape([M, M])       
            # Plot
            plots.image_truth_pred_plot(
                image, 
                truth, 
                pred, 
                self.plot_dir+self.tag+"_comp_%d.pdf" % i,
                fig=fig,
                save=(not fig)
            )

        return

class CompareHelper():
    def __init__(self):
        self.model_helpers = []
        self.n_models = 0
        self.model_tags = []
        self.data_hdf5 = None
        self.metadata_json = None
        self.out_dir = ""

    def add_models(self, models, model_dirs):
        if type(model_dirs) == str:
            model_dirs = glob.glob(model_dir)
        if len(models) == 1:
            for model_dir in model_dirs:
                self.add_model(models[0], model_dir)
        elif len(models) != len(model_dirs):
            print("[COMPARE_HELPER] %d models provided for %d summaries"
                  % (len(models), len(model_dirs)))
            print("[COMPARE_HELPER] --> exiting")
            return
        else:
            for i, model_dir in enumerate(model_dirs):
                self.add_model(models[i], model_dir)
        return

    def add_model(self, model, model_dir, helper=ModelHelper):
        if model_dir[-1] != "/":
            model_dir += "/"
        model_helper = helper(model, model_dir)
        if not self.data_hdf5:
            # Load data and metadata
            self.data_hdf5 = model_helper.data_hdf5
            self.metadata_json = model_helper.metadata_json
            self.data = h5py.File(self.data_hdf5)
            with open(self.metadata_json, "r") as f_in:
                self.metadata = json.load(f_in)
        elif self.data_hdf5 != model_helper.data_hdf5:
            print("[COMPARE_HELPER] Model %s was trained on a different dataset"
                  % model_helper.tag)
            print("[COMPARE_HELPER] --> skipping")
            return
        elif model_helper.tag in self.model_tags:
            print("[COMPARE_HELPER] Model %s already loaded" % model_helper.tag)
            print("[COMPARE_HELPER] --> skipping")
            return
        # Store model
        print("[COMPARE_HELPER] Loaded model %s" % model_helper.tag)
        model_helper.assign_data(self.data, self.metadata)
        self.model_helpers.append(model_helper)
        self.model_tags.append(model_helper.tag)
        self.n_models += 1
        return

    def get_model(self, tag):
        for model_helper in model_helpers:
            if tag == model_helper.tag:
                return model_helper.model

    def compare(self):
        # Set up directories
        self.organize()
        # Check for models
        if not self.model_helpers:
            print("[COMPARE_HELPER] No models loaded")
            return
        else:
            print("[COMPARE_HELPER] Comparing loaded models")
        # Get list of plotting functions
        plot_func_names = []
        for name in dir(ModelHelper):
            attr = getattr(ModelHelper, name)
            if callable(attr) and name.startswith("plot_"):
                plot_func_names.append(name)
        # Loop over models, make individual plots
        for name in plot_func_names:
            print("[COMPARE_HELPER] Running "+name)
            plot_name = name.split("plot_")[-1]
            common_fig = plt.figure()
            for i, model_helper in enumerate(self.model_helpers):
                plot_func = getattr(model_helper, name)
                # Individual plot
                plot_func()
                # Comparison plot
                plot_func(fig=common_fig)

            common_fig.savefig(self.out_dir+plot_name)
            plt.close(common_fig)

    def organize(self):
        # Create base directory
        if not os.path.isdir("trained_models"):
            os.mkdir("trained_models")
        self.out_dir = "trained_models/comparisons/"
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        # Make output directory name
        self.out_dir += "_vs_".join([m.tag for m in self.model_helpers[:2]])
        if self.n_models > 2:
            self.out_dir += "_vs_%d_more" % (self.n_models - 2)
        # Create comparison directory
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)
        # Add a timestamp
        self.out_dir += "/"+dt.today().strftime("%Y-%m-%d")+"/"
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        return


if __name__ == "__main__":
    helper = CompareHelper()
    helper.add_model(models.unet, "trained_models/2p5_1extra_test")
    helper.add_model(models.unet, "trained_models/2p5_2extra_test")
    helper.compare()
