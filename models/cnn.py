import tensorflow.keras as keras

def cnn(n_pixels,config):
    input_img = keras.layers.Input(shape = (n_pixels,n_pixels,n_pixels,1), name = 'input_img')
    dropout = 0.25
    conv = keras.layers.Conv3D(32,kernel_size = (3,3,3), kernel_initializer = 'uniform',
        activation='relu', name = 'conv_1')(input_img)
    conv = keras.layers.MaxPooling3D(pool_size = (2,2,2),name='conv_maxpool_1')(conv)
    conv = keras.layers.Dropout(
        dropout,
        name='conv_dropout_1'
    )(conv)
    conv = keras.layers.Conv3D(
        32,
        kernel_size = (3,3,3),
        kernel_initializer = 'uniform',
        activation = 'relu',
        name = 'conv_2'
    )(conv)
    conv = keras.layers.MaxPooling3D(
        pool_size = (2,2,2),
        name = 'conv_maxpool_2'
    )(conv)
    conv = keras.layers.Dropout(
        dropout,
        name = 'conv_dropout_2'
    )(conv)

    conv = keras.layers.Flatten()(conv)

    output = keras.layers.Dense(
        1,
        activation ='sigmoid',
        kernel_initializer ='lecun_uniform',
        name = 'output'
    )(conv)

    model = keras.models.Model(inputs = [input_img], outputs = [output])
    optimizer = keras.optimizers.Adam()
    model.compile(optimizer = optimizer, loss = 'binary_crossentropy')
    print(model.summary())

    return model

