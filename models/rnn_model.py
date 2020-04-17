#!/usr/bin/env python

import argparse
import glob
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

    timesteps = 5

    # select number of samples for the training
    num_samples_train = x_train.shape[0] / timesteps
    while not num_samples_train.is_integer():
        x_train = x_train[:-1, :]
        y_train = y_train[:-1, :]
        num_samples_train = x_train.shape[0] / timesteps
    num_samples_train = int(num_samples_train)

    # select number of samples for the validation
    num_samples_val = x_val.shape[0] / timesteps
    while not num_samples_val.is_integer():
        x_val = x_val[:-1, :]
        y_val = y_val[:-1, :]
        num_samples_val = x_val.shape[0] / timesteps
    num_samples_val = int(num_samples_val)

    # correctly split the data according to the dimensions calculated
    x_train = np.reshape(
        x_train, (num_samples_train, timesteps, x_train.shape[1]))
    x_val = np.reshape(x_val, (num_samples_val, timesteps, x_val.shape[1]))

    y_train = np.reshape(
        y_train, (num_samples_train, timesteps, y_train.shape[1]))
    y_val = np.reshape(y_val, (num_samples_val, timesteps, y_val.shape[1]))

    print(x_train.shape)
    print(x_val.shape)
    print(y_train.shape)
    print(y_val.shape)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(30,
                                   return_sequences=True,
                                   input_shape=(timesteps, X.shape[1]),
                                   activation='tanh'))
    model.add(tf.keras.layers.LSTM(30, return_sequences=True,
                                   input_shape=(timesteps, X.shape[1]), activation='tanh'))
    model.add(tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(Y.shape[1], activation=None)))

    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(
        # loss=gaussian_nll,
        loss='mean_squared_error',
        optimizer=optimizer,
        #   metrics=[gaussian_mse, gaussian_mae]
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
                'rnn_' + str(epoch) + '_model.h5')))

    # model.fit(x_train, y_train,
    #             batch_size=args.batch_size,
    #             epochs=args.epochs,
    #             verbose=1,
    #             validation_data=(x_val, y_val)
    #             )

    model.save(str(Path(args.path).joinpath('rnn_model.h5')))
