import os, sys

import numpy
import h5py

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import models
import utils

import tensorflow as tf
import tensorflow.keras as keras

f = h5py.File("covid_data.hdf5", "r")

image_features = numpy.array(f['features'])
image_features = image_features.reshape([-1,128,128,1])
label = numpy.array(f['labels'])
label = label.reshape([-1,128,128,1])

X_train, X_test, y_train, y_test = train_test_split(image_features, label, train_size = 0.7)

print(X_train.shape)
print(y_train.shape)
print(len(image_features[0]))

config = {
    "n_pixels" : len(image_features[0]),
    "n_filters" : 24,
    "n_layers_conv" : 1,
    "n_layers_unet" : 3,
    "kernel_size" : 3,
    "dropout" : 0.1,
    "batch_norm" : True,
}

model = models.unet(config) 
#model = models.cnn(len(image_features[0]), config)


savename = "test"
weights_file = "weights/"+savename+"_weights_{epoch:02d}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(weights_file) # save after every epoch 
callbacks_list = [checkpoint]

model.fit([X_train], y_train, epochs = 30, batch_size = 16, callbacks=callbacks_list, validation_data = (X_test, y_test))
model.fit([X_train], y_train, epochs = 20, batch_size = 64, callbacks=callbacks_list, validation_data = (X_test, y_test))

pred = model.predict([X_test], batch_size = 64)
pred_train = model.predict([X_train], batch_size = 64)

n_plots =105
idx = numpy.random.randint(0, len(X_train), n_plots)
for i in idx:
    utils.plot_pneumonia_heatmap(X_train[i], pred_train[i], name = "train_%d" % i)

idx = numpy.random.randint(0, len(X_test), n_plots)
for i in idx:
    utils.plot_pneumonia_heatmap(X_test[i], pred[i], name = "test_%d" % i)

