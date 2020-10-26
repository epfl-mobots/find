#!/usr/bin/env python

import tqdm
import glob
import argparse
from pathlib import Path

import tensorflow as tf

from utils.losses import *


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
        n_dims = int(int(y_pred.shape[3]) / 2)
        mu = y_pred[:, 0, i, :n_dims]
        logsigma = y_pred[:, 0, i, n_dims:]

        max_logvar = 0
        min_logvar = -10
        logsigma = max_logvar - K.log(K.exp(max_logvar - logsigma) + 1)
        logsigma = min_logvar + K.log(K.exp(logsigma - min_logvar) + 1)

        # https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf
        f = -0.5 * \
            K.sum(K.square((y_true[:, 0, i, :] - mu) /
                           K.exp(logsigma)), axis=1)
        sigma_trace = -K.sum(logsigma, axis=1)
        log2pi = -0.5 * n_dims * np.log(2 * np.pi)
        log_likelihood = f + sigma_trace + log2pi
        means.append(K.mean(-log_likelihood))

    return sum(means) / len(means)


def load(exp_path, fname):
    files = glob.glob(exp_path + '/*' + fname)
    pos = []
    vel = []
    for f in files:
        matrix = np.loadtxt(f)
        pos.append(matrix)
        matrix = np.loadtxt(f.replace('positions', 'velocities'))
        vel.append(matrix)
    return pos, vel, files


def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif


def split_cart(data, timestep, args={'center': (0, 0)}):
    if 'center' not in args.keys():
        args['center'] = (0, 0)

    pos = data['pos']
    vel = data['vel']

    offset = 1
    if args['timesteps-skip'] > 0:
        offset = args['timesteps-skip']

    input_list = []
    output_list = []
    for file_idx, p in enumerate(pos):
        inputs = None
        outputs = None

        pos_t_1 = np.roll(p, shift=1, axis=0)[
            1:-offset, :]
        pos_t = p[offset:-1, :]

        vel_t = (pos_t - pos_t_1) / timestep
        vel_t_1 = np.roll(vel_t, shift=1, axis=0)

        pos_t_1 = pos_t_1[1:-1, :]
        vel_t_1 = vel_t_1[1:-1, :]
        pos_t = pos_t[1:-1, :]
        vel_t = vel_t[1:-1, :]

        for fidx in range(p.shape[1] // 2):
            X = []
            Y = []

            X.append(pos_t_1[:, fidx * 2])
            X.append(pos_t_1[:, fidx * 2 + 1])
            X.append(vel_t_1[:, fidx * 2])
            X.append(vel_t_1[:, fidx * 2 + 1])

            Y.append(vel_t[:, fidx * 2] - vel_t_1[:, fidx * 2])
            Y.append(vel_t[:, fidx * 2 + 1] - vel_t_1[:, fidx * 2 + 1])

            for nidx in range(p.shape[1] // 2):
                if fidx == nidx:
                    continue
                X.append(pos_t_1[:, nidx * 2])
                X.append(pos_t_1[:, nidx * 2 + 1])
                X.append(vel_t_1[:, nidx * 2])
                X.append(vel_t_1[:, nidx * 2 + 1])

            if inputs is None:
                inputs = X
                outputs = Y
            else:
                inputs = np.append(inputs, X, axis=1)
                outputs = np.append(outputs, Y, axis=1)
        input_list.append(inputs.T)
        output_list.append(outputs.T)
    return input_list, output_list


def split_data(data, timestep, split_func=split_cart, args={}):
    return split_func(data, timestep, args)


def ready_data(data, timesteps, prediction_steps, args):
    def split(x, y, timesteps, prediction_steps, args):
        X = np.empty([0, timesteps, x.shape[1]])
        if args.prediction_steps == 1:
            Y = np.empty([0, y.shape[1]])
        else:
            Y = np.empty([0, 1, prediction_steps, y.shape[1]])

        iters = 1
        if args.timesteps_skip > 0:
            iters = args.timesteps_skip

        for idxskip in range(iters):
            xh = x[idxskip::(args.timesteps_skip + 1)].copy()
            yh = y[idxskip::(args.timesteps_skip + 1)].copy()

            for i in range(timesteps, xh.shape[0] - prediction_steps):
                inp = xh[(i-timesteps):i, :].reshape(1, timesteps, xh.shape[1])

                if args.prediction_steps == 1:
                    out = yh[i-1, :]
                else:
                    out = yh[(i-1):(i-1+prediction_steps), :].reshape(1,
                                                                      1, prediction_steps, yh.shape[1])
                X = np.vstack((X, inp))
                Y = np.vstack((Y, out))
        return X, Y

    return split(*data, timesteps, prediction_steps, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RNN model to reproduce fish motion')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--epochs', '-e', type=int,
                        help='Number of training epochs',
                        default=1000)
    parser.add_argument('--batch_size', '-b', type=int,
                        help='Batch size',
                        default=256)
    parser.add_argument('--prediction-steps', type=int,
                        help='Trajectory steps to predict',
                        default=1)
    parser.add_argument('--dump', '-d', type=int,
                        help='Batch size',
                        default=100)
    parser.add_argument('--num-timesteps', type=int,
                        help='Number of LSTM timesteps',
                        default=5)
    parser.add_argument('--load', '-l', type=str,
                        help='Load model from existing file and continue the training process',
                        required=False)
    parser.add_argument('--polar', action='store_true',
                        help='Use polar inputs instead of cartesian coordinates',
                        default=False)
    parser.add_argument('--timesteps-skip', type=int,
                        help='Timesteps skipped between input and prediction',
                        default=0,
                        required=False)
    args = parser.parse_args()

    pos, vel, files = load(args.path, 'positions.dat')
    data = {
        'pos': pos,
        'vel': vel,
    }
    if not args.polar:
        split_args = {
            'timesteps-skip': args.timesteps_skip
        }
        X_list, Y_list = split_data(data, args.timestep, args=split_args)
    else:
        # X, Y = split_data(data, args.timestep, split_polar)
        pass

    timesteps = args.num_timesteps

    x_shape = (0, timesteps, X_list[0].shape[1])
    if args.prediction_steps == 1:
        y_shape = (0, Y_list[0].shape[1])
    else:
        y_shape = (0, 1, args.prediction_steps, Y_list[0].shape[1])

    Xh = np.empty(x_shape)
    Yh = np.empty(y_shape)
    for idx in tqdm.tqdm(range(len(X_list))):
        Xi = X_list[idx]
        Yi = Y_list[idx]
        (Xi, Yi) = ready_data((Xi, Yi), timesteps, args.prediction_steps, args)
        if Xi.shape[0] == 0:
            continue
        Xh = np.vstack((Xh, Xi))
        Yh = np.vstack((Yh, Yi))

    split_at = Xh.shape[0] - Xh.shape[0] // 10
    (x_train, x_val) = Xh[:split_at, :, :], Xh[split_at:, :, :]
    (y_train, y_val) = Yh[:split_at, :], Yh[split_at:, :]

    optimizer = tf.keras.optimizers.Adam(0.0001)

    init_epoch = 0
    if args.load:
        model = tf.keras.models.load_model(Path(args.load), custom_objects={
            'Yh': np.empty((0, 2)),
            'multi_dim_gaussian_nll': multi_dim_gaussian_nll,
            'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})

        ints = [int(s) for s in args.load.split('_') if s.isdigit()]
        init_epoch = ints[-1]
    else:
        model = tf.keras.Sequential()

        if args.prediction_steps == 1:
            model.add(tf.keras.layers.LSTM(128, return_sequences=True,
                                           input_shape=(timesteps, Xh.shape[2]), activation='tanh'))
            model.add(tf.keras.layers.Conv1D(
                128, kernel_size=3, input_shape=(100, 1), padding='causal', activation='relu'))
            model.add(tf.keras.layers.MaxPool1D(pool_size=2))
            model.add(tf.keras.layers.Conv1D(
                64, kernel_size=2, padding='causal', activation='relu'))
            model.add(tf.keras.layers.MaxPool1D(pool_size=2))
            model.add(tf.keras.layers.Conv1D(
                32, kernel_size=1, padding='causal', activation='relu'))
            model.add(tf.keras.layers.Flatten())
            model.add(tf.keras.layers.Dense(30, activation='tanh'))
            model.add(tf.keras.layers.Dense(Yh.shape[1] * 2, activation=None))
            model.compile(
                loss=gaussian_nll,
                optimizer=optimizer,
                metrics=[gaussian_mse, gaussian_mae]
            )
        else:
            model.add(tf.keras.layers.LSTM(30, return_sequences=False,
                                           input_shape=(timesteps, Xh.shape[2]), activation='tanh'))
            model.add(tf.keras.layers.Dense(
                Yh.shape[1] * args.prediction_steps * 2, activation=None))
            model.add(tf.keras.layers.Lambda(
                lambda x: tf.reshape(x, shape=(-1, 1, args.prediction_steps, Yh.shape[1] * 2))))
            model.compile(
                loss=multi_dim_gaussian_nll,
                optimizer=optimizer,
            )

    model.summary()

    for epoch in range(init_epoch, args.epochs):
        model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=epoch + 1,
                  initial_epoch=epoch,
                  validation_data=(x_val, y_val),
                  verbose=1)

        if epoch % args.dump == 0:
            model.save(str(Path(args.path).joinpath(
                'prob_lstm_' + str(epoch) + '_model.h5')))

    model.save(str(Path(args.path).joinpath('prob_lstm_model.h5')))
