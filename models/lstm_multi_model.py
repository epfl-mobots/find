#!/usr/bin/env python

import tqdm
import glob
import argparse
from pathlib import Path

import tensorflow as tf

from utils.losses import *


def load(exp_path, fname):
    files = glob.glob(exp_path + '/*' + fname)
    data = []
    for f in files:
        matrix = np.loadtxt(f)
        data.append(matrix)
    return data, files


def split_cartesian(data, timestep, args={'center': (0, 0)}):
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

            # TODO: shuffle or use all combinations of trajectories for input to help the generalization
            # at least in the case of > 2 individuals
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


def split_data(data, timestep, split_func=split_cartesian, args={}):
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
    args = parser.parse_args()

    pos, _ = load(args.path, 'positions.dat')
    data = {
        'pos': pos,
    }
    X, Y = split_data(data, args.timestep)
    X = X.transpose()
    Y = Y.transpose()

    split_at = X.shape[0] - X.shape[0] // 10
    (x_train, x_val) = X[:split_at, :], X[split_at:, :]
    (y_train, y_val) = Y[:split_at, :], Y[split_at:, :]

    timesteps = args.num_timesteps

    ((x_train, y_train), (x_val, y_val)) = ready_data((x_train, y_train), (x_val, y_val),
                                                      timesteps, args.prediction_steps)

    optimizer = tf.keras.optimizers.Adam(0.0001)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(30,
                                   return_sequences=True,
                                   input_shape=(timesteps, X.shape[1]),
                                   activation='tanh'))

    if args.prediction_steps == 1:
        model.add(tf.keras.layers.LSTM(30, return_sequences=False,
                                       input_shape=(timesteps, X.shape[1]), activation='tanh'))
        model.add(tf.keras.layers.Dense(Y.shape[1], activation='tanh'))
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
        )
    else:
        model.add(tf.keras.layers.LSTM(30, return_sequences=False,
                                       input_shape=(timesteps, X.shape[1]), activation='tanh'))
        model.add(tf.keras.layers.Dense(
            Y.shape[1] * args.prediction_steps, activation='tanh'))
        model.add(tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=(-1, 1, args.prediction_steps, Y.shape[1]))))
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
        )

    model.summary()

    for epoch in range(args.epochs):
        model.fit(x_train, y_train,
                  batch_size=args.batch_size,
                  epochs=epoch + 1,
                  initial_epoch=epoch,
                  validation_data=(x_val, y_val),
                  verbose=1)

        if epoch % args.dump == 0:
            model.save(str(Path(args.path).joinpath(
                'lstm_' + str(epoch) + '_model.h5')))

    model.save(str(Path(args.path).joinpath('lstm_model.h5')))
