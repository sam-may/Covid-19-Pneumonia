import os
import numpy
import matplotlib.pyplot as plt
from plots import *
from helpers.compare_helper import CompareHelper
from helpers.model_helper import ModelHelper
from helpers.print_helper import print
from models.cnn import cnn3D as cnn
from models import loss_functions
from generators.cnn import DataGenerator3D

class PlotHelper(ModelHelper):
    def __init__(self, model, model_dir):
        super().__init__(model, model_dir)
        # Loss functions
        self.loss = loss_functions.weighted_crossentropy(self.bce_alpha)

    def assign_data(self, data, metadata):
        self.data = data
        self.metadata = metadata
        self.data_generator = DataGenerator3D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=4,
            input_reshape=(64, 64, 64),
            extra_features=self.extra_features
        )
        return

if __name__ == "__main__":
    # Initialize comparison framework
    basedir="/mnt/data/LungNodules/"
    compare_helper = CompareHelper(
        data_hdf5=basedir+"features_rotatable.hdf5", 
        metadata_json=basedir+"features_rotatable_metadata.json"
    )
    # Initialize plotting functions
    model1_helper = PlotHelper(
        cnn, 
        model_dir="trained_models/nodulesCNN3D_extra-features_azim-rotations"
    )
    model2_helper = PlotHelper(
        cnn, 
        model_dir="trained_models/nodulesCNN3D_extra-features"
    )
    # Add to comparisons list
    compare_helper.add_model(model1_helper)
    compare_helper.add_model(model2_helper)
    # Run comparisons
    compare_helper.compare()
