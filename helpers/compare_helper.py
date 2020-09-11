import os
import json
import glob
import h5py
import matplotlib.pyplot as plt
from datetime import datetime as dt
from .print_helper import print

class CompareHelper():
    def __init__(self, data_hdf5=None, metadata_json=None):
        self.model_helpers = []
        self.n_models = 0
        self.model_tags = []
        self.data = h5py.File(data_hdf5, "r")
        with open(metadata_json, "r") as f_in:
            self.metadata = json.load(f_in)
        self.out_dir = ""

    def add_models(self, model_helpers):
        for model_helper in model_helpers:
            self.add_model(model_helper)

    def add_model(self, model_helper):
        print("Loaded model %s" % model_helper.tag)
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
            print("No models loaded")
            return
        else:
            print("Comparing loaded models")
        # Get list of plotting functions
        plot_func_names = []
        model_helper_class = type(self.model_helpers[0])
        for name in dir(model_helper_class):
            attr = getattr(model_helper_class, name)
            if callable(attr) and name.startswith("plot_"):
                plot_func_names.append(name)
        # Loop over models, make individual plots
        for name in plot_func_names:
            print("Running "+name)
            plot_name = name.split("plot_")[-1]
            common_fig, _ = plt.subplots()
            for i, model_helper in enumerate(self.model_helpers):
                plot_func = getattr(model_helper, name)
                # Individual plot
                plot_func()
                # Comparison plot
                plot_func(fig=common_fig)
            # Check to see if a comparison was plotted
            no_common_plot = True
            common_axes = common_fig.axes
            for axes in common_axes:
                if axes.lines or axes.collections or axes.patches:
                    no_common_plot = False
            if not no_common_plot:
                common_fig.savefig(self.out_dir+plot_name+".pdf")
            # Cleanup
            plt.close(common_fig)

        return

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
        self.out_dir += "/"
        if not os.path.isdir(self.out_dir):
            os.mkdir(self.out_dir)

        return
