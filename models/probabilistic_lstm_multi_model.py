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

    inputs = None
    outputs = None
    for p in pos:
        for fidx in range(p.shape[1] // 2):
            X = []
            Y = []

            pos_t = np.roll(p, shift=1, axis=0)[2:, :]
            pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
            vel_t = (p - np.roll(p, shift=1, axis=0))[2:, :] / timestep
            vel_t_1 = (p - np.roll(p, shift=1, axis=0))[1:-1, :] / timestep

            X.append(pos_t_1[:, fidx * 2])
            X.append(pos_t_1[:, fidx * 2 + 1])
            X.append(vel_t_1[:, fidx * 2])
            X.append(vel_t_1[:, fidx * 2 + 1])

            Y.append(vel_t[:, fidx * 2])
            Y.append(vel_t[:, fidx * 2 + 1])

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
    return inputs, outputs


def split_polar(data, timestep, args={'center': (0, 0)}):
    if 'center' not in args.keys():
        args['center'] = (0, 0)

    inputs = None
    outputs = None
    for idx, f in enumerate(files):
        p = data['pos'][idx]
        v = data['vel'][idx]
        assert p.shape == v.shape, 'Dimensions don\'t match'

        pos_t = np.roll(p, shift=1, axis=0)[2:, :]
        pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
        vel_t = np.roll(v, shift=1, axis=0)[2:, :]
        vel_t_1 = np.roll(v, shift=1, axis=0)[1:-1, :]

        for fidx in range(p.shape[1] // 2):
            X = []
            Y = []

            rad_t_1 = np.sqrt((pos_t_1[:, fidx * 2] - args['center'][0])
                              ** 2 + (pos_t_1[:, fidx * 2 + 1] - args['center'][1]) ** 2)
            hdg_t_1 = np.array(list(map(angle_to_pipi, np.arctan2(
                vel_t_1[:, fidx * 2 + 1], vel_t_1[:, fidx * 2]))))

            X.append(rad_t_1)
            X.append(np.cos(hdg_t_1))
            X.append(np.sin(hdg_t_1))
            X.append(vel_t_1[:, fidx * 2])
            X.append(vel_t_1[:, fidx * 2 + 1])

            Y.append(vel_t[:, fidx * 2])
            Y.append(vel_t[:, fidx * 2 + 1])

            for nidx in range(p.shape[1] // 2):
                if fidx == nidx:
                    continue
                rad_t_1 = np.sqrt((pos_t_1[:, nidx * 2] - args['center'][0])
                                  ** 2 + (pos_t_1[:, nidx * 2 + 1] - args['center'][1]) ** 2)
                hdg_t_1 = np.array(list(map(angle_to_pipi, np.arctan2(
                    vel_t_1[:, nidx * 2 + 1], vel_t_1[:, nidx * 2]))))
                dist_t_1 = np.sqrt((pos_t_1[:, fidx * 2] - pos_t_1[:, nidx * 2]) ** 2 + (
                    pos_t_1[:, fidx * 2 + 1] - pos_t_1[:, nidx * 2 + 1]) ** 2)

                X.append(rad_t_1)
                X.append(np.cos(hdg_t_1))
                X.append(np.sin(hdg_t_1))
                X.append(vel_t_1[:, nidx * 2])
                X.append(vel_t_1[:, nidx * 2 + 1])
                X.append(dist_t_1)

            if inputs is None:
                inputs = X
                outputs = Y
            else:
                inputs = np.append(inputs, X, axis=1)
                outputs = np.append(outputs, Y, axis=1)
    return inputs, outputs


def split_data(data, timestep, split_func=split_cart, args={}):
    return split_func(data, timestep, args)


def ready_data(train_data, validation_data, timesteps, prediction_steps):
    def split(x, y, timesteps, prediction_steps):
        X = np.empty([0, timesteps, x.shape[1]])
        if args.prediction_steps == 1:
            Y = np.empty([0, y.shape[1]])
        else:
            Y = np.empty([0, 1, prediction_steps, y.shape[1]])

        for i in tqdm.tqdm(range(timesteps, x.shape[0] - prediction_steps)):
            inp = x[(i-timesteps):i, :].reshape(1, timesteps, x.shape[1])

            if args.prediction_steps == 1:
                out = y[i-1, :]
            else:
                out = y[(i-1):(i-1+prediction_steps), :].reshape(1,
                                                                 1, prediction_steps, y.shape[1])

            X = np.vstack((X, inp))
            Y = np.vstack((Y, out))

        return X, Y

    return (split(*train_data, timesteps, prediction_steps), split(*validation_data, timesteps, prediction_steps))


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
    args = parser.parse_args()

    pos, vel, files = load(args.path, 'positions.dat')
    data = {
        'pos': pos,
        'vel': vel,
    }
    if not args.polar:
        X, Y = split_data(data, args.timestep)
    else:
        X, Y = split_data(data, args.timestep, split_polar)
    X = X.transpose()
    Y = Y.transpose()

    split_at = X.shape[0] - X.shape[0] // 10
    (x_train, x_val) = X[:split_at, :], X[split_at:, :]
    (y_train, y_val) = Y[:split_at, :], Y[split_at:, :]

    timesteps = args.num_timesteps

    ((x_train, y_train), (x_val, y_val)) = ready_data((x_train, y_train), (x_val, y_val),
                                                      timesteps, args.prediction_steps)

    optimizer = tf.keras.optimizers.Adam(0.0001)

    init_epoch = 0
    if args.load:
        model = tf.keras.models.load_model(Path(args.load), custom_objects={
            'Y': np.empty((0, 2)),
            'multi_dim_gaussian_nll': multi_dim_gaussian_nll,
            'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})

        ints = [int(s) for s in args.load.split('_') if s.isdigit()]
        init_epoch = ints[0]
    else:
        model = tf.keras.Sequential()

        if args.prediction_steps == 1:
            model.add(tf.keras.layers.LSTM(100, return_sequences=False,
                                           input_shape=(timesteps, X.shape[1]), activation='tanh'))
            model.add(tf.keras.layers.Dense(80, activation='tanh'))
            model.add(tf.keras.layers.Dense(50, activation='tanh'))
            model.add(tf.keras.layers.Dense(80, activation='tanh'))
            model.add(tf.keras.layers.Dense(20, activation='tanh'))
            model.add(tf.keras.layers.Dense(Y.shape[1] * 2, activation=None))
            model.compile(
                loss=gaussian_nll,
                optimizer=optimizer,
                metrics=[gaussian_mse, gaussian_mae]
            )
        else:
            model.add(tf.keras.layers.LSTM(30, return_sequences=False,
                                           input_shape=(timesteps, X.shape[1]), activation='tanh'))
            model.add(tf.keras.layers.Dense(
                Y.shape[1] * args.prediction_steps * 2, activation=None))
            model.add(tf.keras.layers.Lambda(
                lambda x: tf.reshape(x, shape=(-1, 1, args.prediction_steps, Y.shape[1] * 2))))
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
