#!/usr/bin/env python

import argparse
import glob
from pathlib import Path

import tensorflow as tf

from losses import *


def load(exp_path, fname):
    files = glob.glob(exp_path + '/*' + fname)
    data = []
    for f in files:
        matrix = np.loadtxt(f)
        data.append(matrix)
    return data, files


def split_polar(data, timestep, args={'center': (0, 0)}):
    if 'center' not in args.keys():
        args['center'] = (0, 0)

    pos = data['pos']

    inputs = None
    outputs = None
    for p in pos:
        for n in range(p.shape[1] // 2):
            pos_t = np.roll(p, shift=1, axis=0)[2:, :]
            pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
            vel_t = (p - np.roll(p, shift=1, axis=0))[2:, :] / timestep
            vel_t_1 = (p - np.roll(p, shift=1, axis=0))[1:-1, :] / timestep

            X = np.array([pos_t_1[:, 0], pos_t_1[:, 1],
                          vel_t_1[:, 0], vel_t_1[:, 1]])
            Y = np.array([vel_t[:, 0], vel_t[:, 1]])
            if inputs is None:
                inputs = X
                outputs = Y
            else:
                inputs = np.append(inputs, X, axis=1)
                outputs = np.append(outputs, Y, axis=1)
    return inputs, outputs


def split_data(data, timestep, split_func=split_polar, args={}):
    return split_func(data, timestep, args)


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
    parser.add_argument('--dump', '-d', type=int,
                        help='Batch size',
                        default=100)
    args = parser.parse_args()

    pos, _ = load(args.path, 'positions_filtered.dat')
    data = {
        'pos': pos,
    }
    X, Y = split_data(data, args.timestep)
    X = X.transpose()
    Y = Y.transpose()

    split_at = X.shape[0] - X.shape[0] // 10
    (x_train, x_val) = X[:split_at, :], X[split_at:, :]
    (y_train, y_val) = Y[:split_at, :], Y[split_at:, :]

    hidden_size = 64
    timesteps = 1

    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_val = np.reshape(x_val, (x_val.shape[0], 1, x_val.shape[1]))

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(30, return_sequences=True,
                                   input_shape=(timesteps, X.shape[1])))
    model.add(tf.keras.layers.LSTM(20, return_sequences=False,
                                   input_shape=(timesteps, X.shape[1])))
    #    model.add(tf.keras.layers.LSTM(30, return_sequences=False,
    #                                   input_shape=(timesteps, X.shape[1])))
    model.add(tf.keras.layers.Dense(Y.shape[1] * 2, activation=None))
    # model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(y_train.shape[1], activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(loss=gaussian_nll,
                  optimizer=optimizer,
                  metrics=[gaussian_mse, gaussian_mae])
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
                'rnn_' + str(epoch) + '_model.h5')))

    # model.fit(x_train, y_train,
    #             batch_size=args.batch_size,
    #             epochs=args.epochs,
    #             verbose=1,
    #             validation_data=(x_val, y_val)
    #             )

    model.save(str(Path(args.path).joinpath('rnn_model.h5')))