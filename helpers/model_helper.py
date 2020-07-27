import json
import glob
import pandas

class ModelHelper():
    def __init__(self, model, model_dir):
        if model_dir[-1] != "/":
            model_dir += "/"
        self.model_dir = model_dir
        self.plot_dir = model_dir+"plots/"
        summary_json = glob.glob(model_dir+"*summary.json")[0]
        with open(summary_json, "r") as f_in:
            summary = json.load(f_in)
            # Load model
            self.model = model(summary["model_config"], verbose=False)
            # Load all training parameters
            for name, value in summary["train_params"].items():
                setattr(self, name, value)
            self.n_extra_slices = int((self.input_shape[-1] - 1)/2.0)
            # External files
            metrics_file = self.model_dir+self.tag+"_metrics.pickle"
            weights_file = (self.model_dir
                            + "weights/"
                            + self.tag
                            + "_weights_01.hdf5")
            self.metrics_df = pandas.read_pickle(metrics_file)
            self.model.load_weights(weights_file)
            # Other
            self.patients_test = summary["patients_test"]
            self.random_seeds = summary["random_seeds"]

    def assign_data(self):
        """
        Must be overridden. Set the following variables:
        
        self.data: input data from hdf5 file
        self.metadata: json with metadata for the above
        self.data_generator: keras generator for generating testing data
        """
        raise NotImplementedError
