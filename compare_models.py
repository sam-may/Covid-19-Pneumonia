import numpy
import matplotlib.pyplot as plt
import plots
from helpers.compare_helper import CompareHelper
from helpers.model_helper import ModelHelper
from models.unet import unet2p5D as unet
from models import loss_functions
from generators import DataGenerator2p5D

class PlotHelper(ModelHelper):
    def __init__(self, model, model_dir):
        super().__init__(model, model_dir)
        # Plot-specific attributes
        self.dice_scores = []

    def assign_data(self, data, metadata)
        self.data = data
        self.metadata = metadata
        self.data_generator = DataGenerator2p5D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=self.validation_batch_size
        )
        return

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

    def plot_learning_curve(self, fig=None):
        save = (not fig)
        if not fig:
            fig = plt.figure()
        else:
            plt.figure(fig.number)
        # Plot
        ax1 = fig.add_subplot(111)
        dice_loss = numpy.array(self.metrics["dice_loss"])
        ax1.plot(numpy.arange(len(dice_loss)), dice_loss, label=self.tag)
        # Formatting
        plt.xlabel("Epoch")
        plt.ylabel("1 - Dice Coefficient")
        plt.legend()
        if save:
            plt.savefig(self.plot_dir+self.tag+"learning_curve.pdf")
            plt.close(fig)

        return

if __name__ == "__main__":
    # Initialize comparison framework
    compare_helper = CompareHelper()
    # Initialize plotting functions
    model1_helper = PlotHelper(unet, "trained_models/2p5_0extra")
    model2_helper = PlotHelper(unet, "trained_models/2p5_5extra")
    # Add to comparisons list
    compare_helper.add_model(model1_helper)
    compare_helper.add_model(model2_helper)
    # Run comparisons
    compare_helper.compare()
