import numpy
import tensorflow.keras as keras
from helpers.train_helper import TrainHelper
from models.unet import unet2p5D as unet
from generators import DataGenerator2p5D
from plots import calc_auc

class UNETHelper(TrainHelper):
    def __init__(self):
        super().__init__()
        self.cli.add_argument(
            "--n_extra_slices", 
            help="extra slices above and below input", 
            type=int, 
            default=0
        )
        self.cli.add_argument(
            "--delta",
            help="Percent by which loss must improve",
            type=float,
            default=0.01
        )
        self.cli.add_argument(
            "--early_stopping_rounds",
            help="Percent by which loss must improve",
            type=int,
            default=2
        )
        self.cli.add_argument(
            "--increase_batch",
            help="Increase batch if more than one 'bad' epoch",
            action="store_true",
            default=True
        )
        self.cli.add_argument(
            "--decay_learning_rate",
            help="Decay learning rate if more than one 'bad' epoch",
            action="store_true",
            default=False
        )
        self.cli.add_argument(
            "--dice_smooth",
            help="Smoothing factor to put in num/denom of dice coeff",
            type=float,
            default=1
        )
        self.cli.add_argument(
            "--bce_alpha",
            help="Weight for positive instances in binary x-entropy",
            type=float,
            default=3
        )
        self.cli.add_argument(
            "--loss_function",
            help="Loss function to use during training",
            type=str,
            default="weighted_crossentropy"
        ) 
        self.parse_cli()

    def train(self):
        """Train model with early stopping"""
        # Initialize data generators
        training_generator = DataGenerator2p5D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_train,
            batch_size=self.training_batch_size
        )
        validation_generator = DataGenerator2p5D(
            data=self.data,
            metadata=self.metadata,
            input_shape=self.input_shape,
            patients=self.patients_test,
            batch_size=self.validation_batch_size
        )
        # Write weights to hdf5 each epoch
        checkpoint = keras.callbacks.ModelCheckpoint(self.weights_file)
        callbacks_list = [checkpoint]
        # Training loop
        train_more = True
        epoch_num = 0
        bad_epochs = 0
        while train_more:
            epoch_num += 1
            if self.verbose:
                print("[TRAIN_HELPER] On epoch %d of training model" 
                      % epoch_num)
            # Run training
            results = self.model.fit(
                training_generator,
                callbacks=callbacks_list,
                use_multiprocessing=False,
                validation_data=validation_generator
            )
            # Calculate TPR, FPR, and AUC
            print("[TRAIN_HELPER] Calculating additional metrics")
            y = []
            for i in range(3): # number of validation set batches
                X, y_ = validation_generator.__getitem__(0)
                pred_ = self.model.predict(X)
                y.append(y_)
                if j == 0:
                    pred = pred_
                else:
                    pred = numpy.concatenate([pred, pred_])
            pred = numpy.array(pred)
            y = numpy.array(y)
            fpr, tpr, auc = plots.calc_auc(y.flatten(), pred.flatten())
            results.history["fpr"] = fpr
            results.history["tpr"] = tpr
            results.history["auc"] = auc
            # Update epoch metrics
            print("[TRAIN_HELPER] Saving epoch metrics")
            self.save_metrics(results.history)
            # Calculate % change for early stopping
            val_loss = results.history["val_loss"][0]
            percent_change = ((self.best_loss - val_loss)/val_loss)*100.0
            if (val_loss*(1. + self.delta)) < self.best_loss:
                print("[TRAIN_HELPER] Loss improved by %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, self.best_loss, val_loss))
                print("[TRAIN_HELPER] --> continuing for another epoch")
                self.best_loss = val_loss
                bad_epochs = 0
            else:
                print("[TRAIN_HELPER] Change in loss was %.2f percent (%.3f -> %.3f)" 
                      % (percent_change, self.best_loss, val_loss)) 
                print("[TRAIN_HELPER] --> incrementing bad epochs by 1")
                bad_epochs += 1
            # Handle dynamic batch size and/or learning rate
            if ((self.increase_batch or self.decay_learning_rate) 
                and bad_epochs >= 1): 
                # Increase batch size (decay learning rate as well?)
                if self.training_batch_size*4 <= self.max_batch_size:
                    print("[TRAIN_HELPER] --> Increasing batch size from %d -> %d" 
                          % (self.training_batch_size, self.training_batch_size*4))
                    print("[TRAIN_HELPER] --> resetting bad epochs to 0")
                    print("[TRAIN_HELPER] --> continuing for another epoch")
                    self.training_batch_size *= 4
                    training_generator.batch_size = self.training_batch_size
                    bad_epochs = 0
            # Check for early stopping
            if bad_epochs >= self.early_stopping_rounds:
                print("[TRAIN_HELPER] Number of early stopping rounds (%d) without\
                      improvement in loss of at least %.2f percent exceeded" 
                      % (self.early_stopping_rounds, self.delta*100.))
                print("[TRAIN_HELPER] --> stopping training after %d epochs" 
                      % (epoch_num))
                train_more = False
            # Stop training after epoch cap
            if self.max_epochs > 0 and epoch_num >= self.max_epochs:
                print("[TRAIN_HELPER] Maximum number of training epochs (%d) reached" 
                      % (self.max_epochs))
                print("[TRAIN_HELPER] --> stopped training")
                train_more = False
        return

if __name__ == "__main__":
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
    unet_helper.run_training(model, unet_config)
