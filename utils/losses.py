import numpy as np
import tensorflow.keras.backend as K


def gaussian_nll(y_true, y_pred):
    """
    :brief: Gaussian negative log likelihood loss function for probabilistic network outputs.

    :param y_true: np.array of the values the network needs to predict
    :param y_pred: np.array of the values the network predicted
    :return: float
    """

    n_dims = int(int(y_pred.shape[1]) / 2)
    mu = y_pred[:, :n_dims]
    logsigma = y_pred[:, n_dims:]

    max_logvar = 0
    min_logvar = -10
    logsigma = max_logvar - K.log(K.exp(max_logvar - logsigma) + 1)
    logsigma = min_logvar + K.log(K.exp(logsigma - min_logvar) + 1)

    # https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf
    f = -0.5 * K.sum(K.square((y_true - mu) / K.exp(logsigma)), axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)
    log_likelihood = f + sigma_trace + log2pi
    return K.mean(-log_likelihood)


def gaussian_mae(y_true, y_pred):
    """
    :brief: Custom mean absolute error function for the Gaussian negative log likelihood function.

    :param y_true: np.array of the values the network needs to predict
    :param y_pred: np.array of the values the network predicted
    :return: float
    """

    n_dims = int(int(y_pred.shape[1]) / 2)
    return K.mean(K.abs(y_pred[:, :n_dims] - y_true), axis=-1)


def gaussian_mse(y_true, y_pred):
    """
    :brief: Custom mean squared error function for the Gaussian negative log likelihood function.

    :param y_true: np.array of the values the network needs to
    :param y_pred: np.array of the values the network predicted
    :return: float
    """

    n_dims = int(int(y_pred.shape[1]) / 2)
    return K.mean(K.square(y_pred[:, :n_dims] - y_true), axis=-1)
