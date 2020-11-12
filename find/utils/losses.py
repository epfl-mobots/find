import numpy as np
import tensorflow.keras.backend as K


def logbound(val, max_logvar=0.5, min_logvar=-10, backend=None):
    if backend is None:
        logsigma = max_logvar - np.log(np.exp(max_logvar - val) + 1)
        logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
    elif backend == 'keras':
        logsigma = max_logvar - K.log(K.exp(max_logvar - val) + 1)
        logsigma = min_logvar + K.log(K.exp(logsigma - min_logvar) + 1)
    return logsigma


def gaussian_nll(y_true, y_pred):
    """
    :brief: Gaussian negative log likelihood loss function for probabilistic network outputs.

    :param y_true: np.array of the values the network needs to predict
    :param y_pred: np.array of the values the network predicted
    :return: float
    """

    n_dims = int(int(y_pred.shape[1]) / 2)
    mu = y_pred[:, :n_dims]
    logsigma = logbound(y_pred[:, n_dims:], 0.5, -10, backend='keras')

    # https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf
    f = -0.5 * K.sum(K.square((y_true - mu) / K.exp(logsigma)), axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5 * n_dims * np.log(2 * np.pi)
    log_likelihood = f + sigma_trace + log2pi
    return K.mean(-log_likelihood)


def multi_dim_gaussian_nll(y_true, y_pred):
    """
    :brief: Gaussian negative log likelihood loss function for probabilistic network outputs.

    :param y_true: np.array of the values the network needs to predict
    :param y_pred: np.array of the values the network predicted
    :return: float
    """

    means = []
    prediction_steps = y_pred.shape[2]
    for i in range(prediction_steps):
        n_dims = y_pred.shape[3] // 2
        mu = y_pred[:, 0, i, :n_dims]
        logsigma = logbound(y_pred[:, 0, i, n_dims:],
                            0.5, -10, backend='keras')

        # https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf
        f = -0.5 * \
            K.sum(K.square((y_true[:, 0, i, :] - mu) /
                           K.exp(logsigma)), axis=1)
        sigma_trace = -K.sum(logsigma, axis=1)
        log2pi = -0.5 * n_dims * np.log(2 * np.pi)
        log_likelihood = f + sigma_trace + log2pi
        means.append(K.mean(-log_likelihood))

    return sum(means) / len(means)


def gaussian_mae(y_true, y_pred):
    """
    :brief: Custom mean absolute error function for the Gaussian negative log likelihood function.

    :param y_true: np.array of the values the network needs to predict
    :param y_pred: np.array of the values the network predicted
    :return: float
    """

    n_dims = y_pred.shape[1] // 2
    return K.mean(K.abs(y_pred[:, :n_dims] - y_true), axis=-1)


def gaussian_mse(y_true, y_pred):
    """
    :brief: Custom mean squared error function for the Gaussian negative log likelihood function.

    :param y_true: np.array of the values the network needs to
    :param y_pred: np.array of the values the network predicted
    :return: float
    """

    n_dims = y_pred.shape[1] // 2
    return K.mean(K.square(y_pred[:, :n_dims] - y_true), axis=-1)
