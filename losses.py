import numpy as np
import tensorflow.keras.backend as K


def gaussian_nll(ytrue, ypreds):    
    n_dims = int(int(ypreds.shape[1])/2)
    mu = ypreds[:, :n_dims]
    logsigma = ypreds[:, n_dims:]
    
    mse = -0.5*K.sum(K.square((ytrue-mu)/K.exp(logsigma)),axis=1)
    sigma_trace = -K.sum(logsigma, axis=1)
    log2pi = -0.5*n_dims*np.log(2*np.pi)
    
    log_likelihood = mse+sigma_trace+log2pi

    return K.mean(-log_likelihood)


def gaussian_mae(y_true, y_pred):
    n_dims = int(int(y_pred.shape[1]) / 2)
    return K.mean(K.abs(y_pred[:, :n_dims] - y_true), axis=-1)
