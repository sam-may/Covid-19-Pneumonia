import tensorflow as tf
import tensorflow.keras as keras

import metrics

def cnn(n_pixels, config):
    input_img = keras.layers.Input(shape=(n_pixels, n_pixels,1), name = 'input_img')

    dropout = 0.25

    conv = keras.layers.Conv2D(
        32, 
        kernel_size=(5,5), 
        kernel_initializer='uniform', 
        activation='relu', 
        name='conv_1'
    )(input_img)
    conv = keras.layers.MaxPooling2D(
        pool_size=(2,2), 
        name='conv_maxpool_1'
    )(conv)
    conv = keras.layers.Dropout(
        dropout, 
        name='conv_dropout_1'
    )(conv)
    conv = keras.layers.Conv2D(
        32, 
        kernel_size=(3,3), 
        kernel_initializer='uniform', 
        activation='relu', 
        name='conv_2'
    )(conv)
    conv = keras.layers.MaxPooling2D(
        pool_size=(2,2), 
        name='conv_maxpool_2'
    )(conv)
    conv = keras.layers.Dropout(
        dropout, 
        name='conv_dropout_2'
    )(conv)
    
    conv = keras.layers.Flatten()(conv)
    
    output = keras.layers.Dense(
        1, 
        activation='sigmoid', 
        kernel_initializer='lecun_uniform', 
        name='output'
    )(conv)

    model = keras.models.Model(inputs=[input_img], outputs=[output])
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy')
    print(model.summary())

    return model

def std_conv(name, input_img, n_layers, n_filters, kernel_size, max_pool=2, 
             dropout=0.25, batch_norm=True, activation='elu', conv_dict={}):
    conv = input_img

    for i in range(n_layers):
        conv = keras.layers.Conv3D(
            n_filters, 
            kernel_size=kernel_size,
            activation=activation,
            kernel_initializer='lecun_uniform',
            padding='same',
            kernel_constraint=keras.constraints.max_norm(10),
            name='std_conv_%s_%d' % (name, i)
        )(conv)

        if dropout > 0:
            conv = keras.layers.Dropout(
                dropout, 
                name='std_conv_dropout_%s_%d' % (name, i)
            )(conv)

        if batch_norm:
            conv = keras.layers.BatchNormalization()(conv)

    if conv_dict: 
        # Store the conv before applying max pooling, so it can be 
        # used later to help give higher resolution information in 
        # the up_convs
        conv_dict['std_conv_%s' % name] = conv

    if max_pool >= 2:
        conv = keras.layers.MaxPooling2D(
            pool_size=(max_pool,max_pool), 
            name='std_conv_maxpool_%s' % name
        )(conv)
    
    return conv

def up_conv(name, input_img, n_layers, n_filters, kernel_size, aux_image=None, 
            dropout=0.25, batch_norm=False):

    conv = input_img

    for i in range(n_layers):
        conv = keras.layers.Conv3DTranspose(
            n_filters, 
            kernel_size,
            strides=2,
            padding='same',
            kernel_constraint=keras.constraints.max_norm(10),
            name='up_conv_%s_%d' % (name, i)
        )(conv)

        if dropout > 0:
            conv = keras.layers.Dropout(
                dropout, 
                name='up_conv_dropout_%s_%d' % (name, i)
            )(conv)

        if batch_norm:
            conv = keras.layers.BatchNormalization()(conv)

    if aux_image is not None:
        conv = keras.layers.Concatenate()([conv, aux_image])

    return conv
    
def unet(config):
    # Unpack config
    input_shape = config["input_shape"]
    n_filters = config["n_filters"]
    n_layers_conv = config["n_layers_conv"]
    n_layers_unet = config["n_layers_unet"]
    kernel_size = config["kernel_size"]
    dropout = config["dropout"]
    batch_norm = config["batch_norm"]
    learning_rate = config["learning_rate"]
    alpha = config["alpha"]
    # Set up input
    input_img = keras.layers.Input(shape=input_shape, name='input_img')
    conv = input_img
    conv_dict = {'input_img': conv}
    # Construct UNET
    for i in range(n_layers_unet):
        conv = std_conv(
            str(i), 
            conv, 
            n_layers_conv, 
            n_filters*(2**i), 
            kernel_size, 
            dropout=dropout, 
            conv_dict=conv_dict, 
            batch_norm=batch_norm
        )
    conv = std_conv(
        'bottom_conv', 
        conv, 
        n_layers_conv, 
        n_filters*(2**n_layers_unet), 
        kernel_size, 
        1, 
        dropout=dropout, 
        batch_norm=batch_norm
    )
    for i in range(n_layers_unet):
        conv = up_conv(
            str(i), 
            conv, 
            1, 
            n_filters, 
            kernel_size, 
            conv_dict['std_conv_%d' % (n_layers_unet - (i+1))], 
            dropout=dropout
        )
        conv = std_conv(
            'lateral_%s' % i, 
            conv, 
            n_layers_conv, 
            n_filters*(2**(n_layers_unet - (i+1))), 
            kernel_size, 
            1, 
            dropout=dropout, 
            batch_norm=batch_norm
        )
    # Output layer
    output = std_conv(
        'out', 
        conv, 
        1, 
        1, 
        kernel_size, 
        1, 
        dropout=0, 
        batch_norm=False, 
        activation='sigmoid'
    ) 
    # Put model together
    model = keras.models.Model(inputs=[input_img], outputs=[output])
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(
        optimizer=optimizer, 
        loss=metrics.weighted_crossentropy(alpha), 
        metrics=['accuracy', metrics.dice_loss]
    )

    print(model.summary())

    return model

