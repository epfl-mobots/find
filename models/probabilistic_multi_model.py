#!/usr/bin/env python

import argparse
import glob
from pathlib import Path

import tensorflow as tf

from losses import *


def load(exp_path, fname):
    files = glob.glob(exp_path + '/*' + fname)
    pos = []
    vel = []
    for f in files:
        matrix = np.loadtxt(f)
        pos.append(matrix)
        matrix = np.loadtxt(f.replace('positions', 'velocities').replace('filtered', 'filtered_twice'))
        vel.append(matrix)
    return pos, vel, files


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


def split_data(data, timestep, split_func=split_polar, args={}):
    return split_func(data, timestep, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dense model to reproduce fish motion')
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

    pos, vel, files = load(args.path, 'positions_filtered.dat')
    data = {
        'pos': pos,
        'vel': vel,
        'files': files
    }
    X, Y = split_data(data, args.timestep)
    X = X.transpose()
    Y = Y.transpose()

    split_at = X.shape[0] - X.shape[0] // 10
    (x_train, x_val) = X[:split_at, :], X[split_at:, :]
    (y_train, y_val) = Y[:split_at, :], Y[split_at:, :]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.Dense(50, activation='tanh'))
    model.add(tf.keras.layers.Dense(50, activation='tanh'))
    # model.add(tf.keras.layers.Dense(30, activation='tanh'))
    model.add(tf.keras.layers.Dense(Y.shape[1] * 2, activation=None))

    loss = gaussian_nll
    optimizer = tf.keras.optimizers.Adam(0.0001)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=[gaussian_mse, gaussian_mae]
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
                'prob_' + str(epoch) + '_model.h5')))

    # model.fit(x_train, y_train,
    #             batch_size=args.batch_size,
    #             epochs=args.epochs,
    #             verbose=1,
    #             validation_data=(x_val, y_val)
    #             )

    model.save(str(Path(args.path).joinpath('prob_model.h5')))
