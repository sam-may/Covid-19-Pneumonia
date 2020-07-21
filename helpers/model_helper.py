import json
import glob
import numpy

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

    def assign_data(self):
        """
        Must be overridden. Set the following variables:
        
        self.data: input data from hdf5 file
        self.metadata: json with metadata for the above
        self.data_generator: keras generator for generating testing data
        """
        raise NotImplementedError
