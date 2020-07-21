import tensorflow.keras as keras
from helpers.train_helper import TrainHelper, train_decorator
from models.unet import unet2p5D as unet
from generators import DataGenerator2p5D

class UNETHelper(TrainHelper):
    def __init__(self):
        super().__init__()
        # Initialize data generators
        self.training_generator = DataGenerator2p5D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_train,
            batch_size=self.training_batch_size
        )
        self.validation_generator = DataGenerator2p5D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=self.validation_batch_size
        )

    @train_decorator
    def train(self, model, model_config):
        """Train model with early stopping"""
        # Store model config and model
        self.summary["model_config"] = model_config
        self.model = model
        # Write weights to hdf5 each epoch
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_file)
        callbacks_list = [checkpoint]
        # Training loop
        train_more = True
        self.n_epochs = 0
        self.bad_epochs = 0
        while train_more:
            self.n_epochs += 1
            if self.verbose:
                print("[TRAIN_HELPER] On epoch %d of training model" 
                      % self.n_epochs)
            # Run training
            results = self.model.fit(
                self.training_generator,
                callbacks=callbacks_list,
                use_multiprocessing=False,
                validation_data=self.validation_generator
            )
            # Update epoch metrics
            print("[TRAIN_HELPER] Saving epoch metrics")
            for name in ["loss", "accuracy", "dice_loss"]:
                self.metrics[name+"_train"].append(results.history[name][0])
                self.metrics[name].append(results.history["val_"+name][0])
            # Calculate % change for early stopping
            val_loss = results.history["val_loss"][0]
            percent_change = ((self.best_loss - val_loss)/val_loss)*100.0
            if (val_loss*(1. + self.delta)) < self.best_loss:
                print("[TRAIN_HELPER] Loss improved by %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, self.best_loss, val_loss))
                print("[TRAIN_HELPER] --> continuing for another epoch")
                self.best_loss = val_loss
                self.bad_epochs = 0
            else:
                print("[TRAIN_HELPER] Change in loss was %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, self.best_loss, val_loss)) 
                print("[TRAIN_HELPER] --> incrementing bad epochs by 1")
                self.bad_epochs += 1
            # Handle dynamic batch size and/or learning rate
            if ((self.increase_batch or self.decay_learning_rate) 
                and self.bad_epochs >= 1): 
                # Increase batch size (decay learning rate as well?)
                if self.training_batch_size*4 <= self.max_batch_size:
                    print("[TRAIN_HELPER] --> Increasing batch size from %d -> %d" 
                          % (self.training_batch_size, self.training_batch_size*4))
                    print("[TRAIN_HELPER] --> resetting bad epochs to 0")
                    print("[TRAIN_HELPER] --> continuing for another epoch")
                    self.training_batch_size *= 4
                    self.training_generator.batch_size = self.training_batch_size
                    self.bad_epochs = 0
            # Check for early stopping
            if self.bad_epochs >= self.early_stopping_rounds:
                print("[TRAIN_HELPER] Number of early stopping rounds (%d) without\
                      improvement in loss of at least %.2f percent exceeded" 
                      % (self.early_stopping_rounds, self.delta*100.))
                print("[TRAIN_HELPER] --> stopping training after %d epochs" 
                      % (self.n_epochs))
                train_more = False
            # Stop training after epoch cap
            if self.max_epochs > 0 and self.n_epochs >= self.max_epochs:
                print("[TRAIN_HELPER] Maximum number of training epochs (%d) reached" 
                      % (self.max_epochs))
                print("[TRAIN_HELPER] --> stopped training")
                train_more = False
        return

if __name__ == "__main__":
    import models
    # Initialize helper
    unet_helper = UNETHelper()
    # Initialize model
    unet_config = {
        "input_shape": unet_helper.input_shape,
        "n_filters": 12,
        "n_layers_conv": 2,
        "n_layers_unet": 3,
        "kernel_size": (4, 4),
        "dropout": 0.0,
        "batch_norm": False,
        "learning_rate": 0.00005,
        "bce_alpha": unet_helper.bce_alpha,
        "dice_smooth": unet_helper.dice_smooth,
        "loss_function": unet_helper.loss_function 
    }
    model = unet(unet_config)
    # Train model
    unet_helper.train(model, unet_config)
