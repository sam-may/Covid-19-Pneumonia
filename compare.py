import argparse
import json
import glob
import numpy
import h5py
import tensorflow
import tensorflow.keras as keras
import loss_functions
import plots
import models
from train import DataGenerator

class ModelHelper():
    def __init__(self, model, summary_json):
        self.plot_dir = "plots/"
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
        self.micro_dice_patients = []

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

    def plot_micro_dice(self):
        """Histograms of the scores for 70% of validation cases"""
        # Get case-by-case dice scores
        dice_scores = []
        for patient in self.patients_test:
            n_slices = len(self.metadata[patient])
            for i in range(int(0.7*n_slices)):
                X, y = self.data_generator.get_random_slice(patient=patient)
                y_pred = self.model.predict(numpy.array([X]), batch_size=1)
                dice_scores.append(loss_functions.dice_loss(y, y_pred))

        plots.hist_1D(
            dice_scores, 
            numpy.linspace(0,1,21),
            self.plot_dir+self.tag+"_micro_dice.pdf",
            title="Micro Dice",
            xlabel="Dice Score"
        )

        return

    def plot_roc_curve(self):
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
            self.plot_dir+self.tag+"_roc_curve.pdf"
        )
        return

    def plot_side_by_side(self, n_plots=5): 
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
                self.plot_dir+self.tag+"_comp_%d.pdf" % i
            )

        return

class CompareHelper():
    def __init__(self):
        # Command Line Interface (CLI)
        cli = argparse.ArgumentParser()
        # General
        cli.add_argument("-a", "--all_plots", action="store_true", default=False)
        cli.add_argument("--micro_dice", action="store_true", default=False)
        cli.add_argument("--roc_curve", action="store_true", default=False)
        cli.add_argument("--side_by_side", action="store_true", default=False)
        # Load CLI args into namespace
        cli.parse_args(namespace=self)
        plot_flags = (vars(cli.parse_args())).values()
        if sum(plot_flags) == 0:
            self.all_plots = True
        # Other attributes
        self.model_helpers = []
        self.model_tags = []
        self.data_hdf5 = None
        self.metadata_json = None

    def add_models(self, models, summary_jsons):
        if type(summary_jsons) == str:
            summary_jsons = glob.glob(summary_jsons)
        if len(models) == 1:
            for summary_json in summary_jsons:
                self.add_model(models[0], summary_json)
        elif len(models) != len(summary_jsons):
            print("[COMPARE_HELPER] %d models provided for %d summaries"
                  % (len(models), len(summary_jsons)))
            print("[COMPARE_HELPER] --> exiting")
            return
        else:
            for i, summary_json in enumerate(summary_jsons):
                self.add_model(models[i], summary_json)
        return

    def add_model(self, model, summary_json):
        model_helper = ModelHelper(model, summary_json)
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
        self.model_helpers.append(model_helper)
        self.model_tags.append(model_helper.tag)
        return

    def get_model(self, tag):
        for model_helper in model_helpers:
            if tag == model_helper.tag:
                return model_helper.model

    def compare(self):
        if not self.model_helpers:
            print("[COMPARE_HELPER] No models loaded")
            return
        else:
            print("[COMPARE_HELPER] Comparing loaded models")
        # Loop over models, make individual plots
        for i, model_helper in enumerate(self.model_helpers):
            model_helper.assign_data(self.data, self.metadata)
            if self.micro_dice or self.all_plots:
                model_helper.plot_micro_dice()
            if self.roc_curve or self.all_plots:
                model_helper.plot_roc_curve()
            if self.side_by_side or self.all_plots:
                model_helper.plot_side_by_side()

if __name__ == "__main__":
    helper = CompareHelper()
    helper.add_model(models.unet, "2p5_1extra_test_summary.json")
    helper.add_model(models.unet, "2p5_2extra_test_summary.json")
    helper.compare()
