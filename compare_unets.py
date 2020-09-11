import os
import numpy
import matplotlib.pyplot as plt
from plots import *
from helpers.compare_helper import CompareHelper
from helpers.model_helper import ModelHelper
from helpers.print_helper import print
from models.unet import unet2p5D as unet
from models import loss_functions
from generators.unet import DataGenerator2p5D

class PlotHelper(ModelHelper):
    def __init__(self, model, model_dir, common_slices):
        super().__init__(model, model_dir)
        # Plot-specific attributes
        self.dice_scores = []
        self.common_slices = common_slices
        # Choose first training cohort
        self.patients_test_0 = self.patients_test[0]
        self.metrics_df_0 = self.metrics_df[self.metrics_df.random_seed == 0]
        # Loss functions
        self.dice_loss = loss_functions.dice_loss(self.dice_smooth)

    def assign_data(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.data_generator = DataGenerator2p5D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test_0,
            batch_size=self.validation_batch_size,
            extra_slice_step=self.extra_slice_step
        )
        return

    def plot_common_slices(self, fig=None):
        is_individual_plot = (not fig)
        if is_individual_plot and len(self.common_slices) > 0:
            common_dir = self.plot_dir+"common/"
            if not os.path.exists(common_dir):
                os.mkdir(common_dir)
            print("Plotting common slice plots")
            # Plot
            for patient_slice_pair in self.common_slices:
                patient, slice_idx = patient_slice_pair
                # Explicit casting because numpy casts the tuple to strings
                out_path = common_dir+self.tag+"_%s_%d.pdf" % (patient, slice_idx)
                self.side_by_side_plot(
                    patient=str(patient), 
                    slice_idx=int(slice_idx), 
                    out_path=out_path
                )
        return 

    def plot_micro_dice(self, fig=None):
        """Histograms of the scores for 70% of validation cases"""
        is_individual_plot = (not fig)
        # Get case-by-case dice scores
        dice_scores = self.dice_scores
        patient_slice_pairs = []
        if not dice_scores:
            print("Generating dice scores")
            generator = self.data_generator
            for patient in self.patients_test_0:
                n_slices = len(self.metadata[patient])
                n_to_plot = int(0.25*n_slices)
                batch_size = (100 if n_to_plot >= 100 else 10)
                for i in range(n_to_plot//batch_size):
                    # Get a large batch of data and truth
                    X = []
                    y = []
                    slice_indices = []
                    for i in range(batch_size):
                        X_, y_, idx_ = generator.get_random_slice(
                            patient=patient,
                            return_slice_idx=True
                        )
                        X.append(X_)
                        y.append(y_)
                        slice_indices.append(idx_)
                    # Run model over entire batch
                    y_pred = self.model.predict(
                        numpy.array(X), 
                        batch_size=batch_size
                    )
                    # Get dice loss for each prediction
                    for i, pred in enumerate(y_pred):
                        truth = y[i]
                        slice_idx = slice_indices[i]
                        dice_score = self.dice_loss(truth, pred)
                        dice_scores.append(dice_score)
                        patient_slice_pairs.append((patient, slice_idx))
        # Plot
        general.hist_1D(
            dice_scores, 
            numpy.linspace(0,1,21),
            self.plot_dir+self.tag+"_micro_dice.pdf",
            title="Micro Dice",
            xlabel="1 - Dice Score",
            fig=fig,
            save=(is_individual_plot),
            tag=self.tag
        )
        if is_individual_plot:
            sample_dir = self.plot_dir+"samples/"
            if not os.path.exists(sample_dir):
                os.mkdir(sample_dir)
            print("Plotting slice plots")
            dice_scores = numpy.array(dice_scores)
            patient_slice_pairs = numpy.array(patient_slice_pairs)
            # Sort dice scores in ascending order
            sorted_by_score = numpy.argsort(dice_scores)
            dice_scores = dice_scores[sorted_by_score]
            patient_slice_pairs = patient_slice_pairs[sorted_by_score]
            # Get evenly-spaced (in dice score) patient-slice pairs
            assortment = numpy.linspace(0, len(dice_scores)-1, 10)
            assortment = numpy.round(assortment).astype(int) # Round to integers
            assortment_of_pairs = patient_slice_pairs[assortment.astype(int)]
            # Plot
            for patient, slice_idx in assortment_of_pairs:
                # Explicit casting because numpy casts the tuple to strings
                self.side_by_side_plot(
                    patient=str(patient), 
                    slice_idx=int(slice_idx),
                    out_path=sample_dir
                )

        return
 
    def plot_learning_curve(self, fig=None):
        save = (not fig)
        if not fig:
            fig, axes = plt.subplots()
        else:
            plt.figure(fig.number)
            axes = fig.axes[0]
        # Plot
        dice_loss = self.metrics_df_0.calc_dice_loss
        axes.plot(numpy.arange(len(dice_loss)), dice_loss, label=self.tag)
        # Formatting
        plt.xlabel("Epoch")
        plt.ylabel("1 - Dice Coefficient")
        plt.legend()
        if save:
            plt.savefig(self.plot_dir+self.tag+"_learning_curve.pdf")
            plt.close(fig)

        return

    def side_by_side_plot(self, patient=None, slice_idx=None, out_path=None): 
        """
        Make plots of (orig|truth|pred)\\(orig+truth|orig+pred|original+(pred-truth))
        """
        # Get data
        if not patient or not slice_idx:
            input_data, truth = self.data_generator.get_random_slice(patient=patient)
        else:
            input_data, truth = self.data_generator.get_slice(patient, slice_idx)
        # Run inference
        pred = self.model.predict(numpy.array([input_data]), batch_size=1)
        # Get dice score
        dice_score = self.dice_loss(truth, pred)
        dice = str(numpy.array(dice_score).flatten()[0])
        # Extract slice of interest from input data
        slice_index = self.n_extra_slices
        orig = input_data[:,:,slice_index]
        # Reshape into plottable images
        M = self.input_shape[0]
        image = orig.reshape([M, M])
        truth = truth.reshape([M, M])
        pred = pred.reshape([M, M])       
        # Plot
        if not out_path:
            out_path = self.plot_dir+self.tag+"_%s_%d.pdf" % (patient, slice_idx)
        slices.image_truth_pred_plot(
            image, 
            truth, 
            pred, 
            out_path,
            title="1 - Dice: %0.2f (%s, %d)" % (dice_score, patient, slice_idx)
        )

        return

if __name__ == "__main__":
    # Initialize comparison framework
    basedir = "/public/smay/covid_ct_data/features/14Jul2020_z_score_downsample256" 
    compare_helper = CompareHelper(
        data_hdf5=basedir+"features.hdf5", 
        metadata_json=basedir+"features.json"
    )
    # Common slices to plot
    common_slices = [
        ("patient18", 154),
        ("russia_patient_274", 14),
        ("patient34", 197),
        ("patient58", 154),
        ("patient66", 257),
        ("patient67", 112),
        ("russia_patient_283", 15),
        ("russia_patient_286", 16),
        ("russia_patient_264", 14),
        ("patient45", 56)
    ]
    # Initialize plotting functions
    model1_helper = PlotHelper(
        unet, 
        "batch/trained_models/test_condor", 
        common_slices
    )
    #model2_helper = PlotHelper(
    #    unet, 
    #    "trained_models/2p5_5extra",
    #    common_slices
    #)
    # Add to comparisons list
    compare_helper.add_model(model1_helper)
    #compare_helper.add_model(model2_helper)
    # Run comparisons
    compare_helper.compare()
