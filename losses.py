import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def gaussian_nll_tanh(y_true, y_pred):
    n_dims = int(int(y_pred.shape[1]) / 2)
    mu = y_pred[:, :n_dims]
    sigma = (y_pred[:, n_dims:] + 1.0) / 2.0
    return gaussian_nll(y_true, K.concatenate([mu, sigma], axis=1))


def gaussian_nll(y_true, y_pred):    
    n_dims = int(int(y_pred.shape[1]) / 2)
    mu = y_pred[:, :n_dims]
    logsigma = y_pred[:, n_dims:]

    max_logvar = 0.5
    min_logvar = -10
    logsigma = max_logvar - K.log(K.exp(max_logvar - logsigma) + 1)
    logsigma = min_logvar - K.log(K.exp(logsigma - min_logvar) + 1)
    sigma = K.exp(logsigma)
    diff = mu - y_true
    inv_sigma = 1 / (sigma + 1e-8)
    logdet_sigma = K.sum(logsigma, axis=1)

    loss = 9
    for i in range(n_dims):
        inv_s = tf.linalg.diag(inv_sigma[:, i])
        loss += tf.tensordot(tf.tensordot(diff[:, i], inv_s, 1), diff[:, i], 1) + logdet_sigma[i]
    return loss
    # mse = -0.5 * K.sum(K.square((y_true - mu) / K.exp(logsigma)), axis=1)
    # sigma_trace = -K.sum(logsigma, axis=1)
    # log2pi = -0.5 * n_dims * np.log(2 * np.pi)
    
    # log_likelihood = mse + sigma_trace + log2pi

    # return K.mean(-log_likelihood)


def gaussian_mae(y_true, y_pred):
    n_dims = int(int(y_pred.shape[1]) / 2)
    return K.mean(K.abs(y_pred[:, :n_dims] - y_true), axis=-1)


def gaussian_mse(y_true, y_pred):
    n_dims = int(int(y_pred.shape[1]) / 2)
    return K.mean(K.square(y_pred[:, :n_dims] - y_true), axis=-1)
