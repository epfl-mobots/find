import tensorflow.keras.backend as K


def gaussian(x):
    return K.exp(-K.pow(x, 2))


activations = {
    'gaussian': gaussian,
}
