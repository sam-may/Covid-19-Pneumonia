import tensorflow as tf
import tensorflow.keras as keras

def weighted_crossentropy(alpha):
    def calc_weighted_crossentropy(y_true, y_pred):
        y_pred = keras.backend.clip(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon()) # to prevent nan's in loss
    
        positive_term = alpha * (-y_true * keras.backend.log(y_pred)) # naively set alpha = sum(negative instances) / sum(positive instances)
        negative_term = -(1 - y_true) * keras.backend.log(1 - y_pred)
        return keras.backend.mean(positive_term + negative_term)

    return calc_weighted_crossentropy

def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - (numerator / denominator)
    #return 1 - (numerator + 1) / (denominator + 1)

#def dice_coefficient(y_true, y_pred):
#    return
