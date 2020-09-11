import json
import glob
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from .print_helper import print


class ModelHelper():
    def __init__(self, model, model_dir):
        if model_dir[-1] != "/":
            model_dir += "/"
        self.model_dir = model_dir
        self.plot_dir = model_dir+"plots/"
        with open(self.model_dir+"summary.json", "r") as f_in:
            summary = json.load(f_in)
            # Load model
            self.model = model(summary["model_config"], verbose=False)
            # Load all training parameters
            for name, value in summary["train_params"].items():
                setattr(self, name, value)
            self.n_extra_slices = int((self.input_shape[-1] - 1)/2.0)
            # External files
            metrics_file = self.model_dir+"metrics.pickle"
            self.metrics_df = pandas.read_pickle(metrics_file)
            self.load_weights()
            # Other
            self.patients_test = summary["patients_test"]
            self.random_seeds = summary["random_seeds"]

    def load_weights(self, weights_file=""):
        if not weights_file:
            weights_files = glob.glob(self.model_dir+"weights/*.hdf5")
            best_loss = 999999.
            for f in weights_files:
                # Choose first training
                if "training" in f and "training-00" not in f:
                    continue
                # Return last epoch if no loss in weights file name
                if "loss" not in f:
                    return weights_files[-1]
                # Remove extraneous strings
                f_info = f.split("weights_")[-1].split(".hdf5")[0]
                # Extract loss
                props = dict([pair.split("-") for pair in f_info.split("_")])
                loss = float(props["loss"])
                if loss < best_loss:
                    best_loss = loss
                    weights_file = f
        print("Loading weights from %s" % weights_file)
        self.model.load_weights(weights_file)
        return

    def assign_data(self):
        """
        Must be overridden. Set the following variables:
        
        self.data: input data from hdf5 file
        self.metadata: json with metadata for the above
        self.data_generator: keras generator for generating testing data
        """
        raise NotImplementedError

    @staticmethod
    def roc_plot(fpr_mean, fpr_std, tpr_mean, tpr_std, auc, auc_std, name, 
                 fig=None, save=True, tag=None):
        """
        Plot mean False Positive Rates (FPR) against mean True Positive Rates
        (TPR) with 1 sigma confidence bands
        """
        if not fig:
            fig, axes = plt.subplots()
        else:
            axes = fig.axes[0]
        axes.yaxis.set_ticks_position('both')
        axes.grid(True)
        axes.plot(
            fpr_mean, 
            tpr_mean,
            label="%s [AUC: %.3f +/- %.3f]" % (tag, auc, auc_std)
        )
        axes.fill_between(
            fpr_mean,
            tpr_mean - (tpr_std/2.),
            tpr_mean + (tpr_std/2.),
            alpha=0.25, 
            label=r'$\pm 1\sigma$'
        )
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        if save:
            plt.savefig(name)
            plt.close(fig)

        return

    def plot_roc_curve(self, fig=None):
        """
        Calculate and plot tpr, fpr, auc and uncertainties
        
        Uncertainties are calculated only if at least 3 different 
        test/train splits are saved in the results
        """
        n_trainings = self.metrics_df.random_seed.nunique()
        # Collect ROC curve data
        tprs = []
        fprs = []
        aucs = []
        for i in range(n_trainings):
            this_training = (self.metrics_df.random_seed == i)
            tpr = self.metrics_df.tpr[this_training].to_numpy()[-1]
            fpr = self.metrics_df.fpr[this_training].to_numpy()[-1]
            auc = self.metrics_df.auc[this_training].to_numpy()[-1]
            tprs.append(tpr)
            fprs.append(fpr)
            aucs.append(auc)
        # Calculate averages
        tpr_mean = numpy.mean(tprs, axis=0)
        fpr_mean = numpy.mean(fprs, axis=0)
        auc = numpy.mean(aucs)
        # Get standard deviations
        if n_trainings >= 3:
            tpr_std = numpy.std(tprs, axis=0)
            fpr_std = numpy.std(fprs, axis=0)
            auc_std = numpy.std(aucs)
        else:
            tpr_std = numpy.zeros_like(tpr_mean)
            fpr_std = numpy.zeros_like(fpr_mean)
            auc_std = 0.
        # Plot
        ModelHelper.roc_plot(
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
