import tensorflow.keras as keras
from . import loss_functions

def cnn3D(config, verbose=True):
    # Unpack config
    input_shape = config["input_shape"]
    n_extra_features = config["n_extra_features"]
    dropout = config["dropout"]
    loss_hyperparams = { 
        "bce_alpha": config["bce_alpha"],
        "dice_smooth": config["dice_smooth"]
    }
    # Retrieve loss function
    loss_function = getattr(loss_functions, config["loss_function"])
    # Set up input
    input_img = keras.layers.Input(
        shape=input_shape, 
        name="input_img"
    )
    if n_extra_features > 0:
        input_features = keras.layers.Input(
            shape=(n_extra_features,), 
            name="extra_features"
        )
    # Construct CNN
    conv = keras.layers.Conv3D(
        2, 
        kernel_size=(4,4,4), 
        kernel_initializer="uniform", 
        activation="relu", 
        name="conv_1"
    )(input_img)
    conv = keras.layers.MaxPooling3D(
        pool_size=(2,2,2), 
        name="conv_maxpool_1"
    )(conv)
    conv = keras.layers.Dropout(
        dropout, 
        name="conv_dropout_1"
    )(conv)
    conv = keras.layers.Conv3D(
        2, 
        kernel_size=(2,2,2), 
        kernel_initializer="uniform", 
        activation="relu", 
        name="conv_2"
    )(conv)
    conv = keras.layers.MaxPooling3D(
        pool_size=(2,2,2), 
        name="conv_maxpool_2"
    )(conv)
    conv = keras.layers.Dropout(
        dropout, 
        name="conv_dropout_2"
    )(conv)
    flat = keras.layers.Flatten()(conv)
    # Add extra features to flattened layer
    if n_extra_features > 0:
        flat = keras.layers.concatenate([input_features, flat])
    # Consolidate into a dense layer
    dense = keras.layers.Dense(
        100, 
        activation="relu", 
        kernel_initializer="lecun_uniform", 
        name="dense_1"
    )(flat)
    # Output layer
    output = keras.layers.Dense(
        1, 
        activation="sigmoid", 
        kernel_initializer="lecun_uniform", 
        name="output"
    )(dense)
    # Put model together
    if n_extra_features > 0:
        model = keras.models.Model(
            inputs=[input_img, input_features], 
            outputs=[output]
        )
    else:
        model = keras.models.Model(inputs=[input_img], outputs=[output])
    optimizer = keras.optimizers.Adam()
    model.compile(
        optimizer=optimizer, 
        loss=loss_function(**loss_hyperparams),
        metrics=[
            "accuracy",
            loss_functions.dice_loss(**loss_hyperparams),
            loss_functions.weighted_crossentropy(**loss_hyperparams)
        ]
    )
    if verbose:
        print(model.summary())

    return model
