#!/usr/bin/env python

import os
import argparse
import glob
from pathlib import Path

import tensorflow as tf

from utils.losses import *


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

    inputs = None
    outputs = None
    for idx, f in enumerate(files):
        p = data['pos'][idx]
        v = data['vel'][idx]
        assert p.shape == v.shape, 'Dimensions don\'t match'

        for n in range(p.shape[1] // 2):
            pos_t = np.roll(p, shift=1, axis=0)[2:, :]
            pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
            vel_t = np.roll(v, shift=1, axis=0)[2:, :]
            vel_t_1 = np.roll(v, shift=1, axis=0)[1:-1, :]

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


def split_polar(data, timestep, args={'center': (0, 0)}):
    if 'center' not in args.keys():
        args['center'] = (0, 0)

    pos = data['pos']

    inputs = None
    outputs = None
    for p in pos:
        for n in range(p.shape[1] // 2):
            pos_t = np.roll(p, shift=1, axis=0)[2:, :]
            rad_t = np.sqrt( (pos_t[:, 0] - args['center'][0]) ** 2 + (pos_t[:, 1] - args['center'][1]) ** 2)

            pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
            rad_t_1 = np.sqrt( (pos_t_1[:, 0] - args['center'][0]) ** 2 + (pos_t_1[:, 1] - args['center'][1]) ** 2)

            vel_t = (p - np.roll(p, shift=1, axis=0))[2:, :] / timestep
            radial_vel_t = (pos_t[:, 0] * vel_t[:, 1] - pos_t[:, 1] * vel_t[:, 0]) / (pos_t[:, 0] ** 2 + pos_t[:, 1] ** 2)
            hdg_t = np.array(list(map(angle_to_pipi, np.arctan2(vel_t[:, 1], vel_t[:, 0]))))

            vel_t_1 = (p - np.roll(p, shift=1, axis=0))[1:-1, :] / timestep
            radial_vel_t_1 = (pos_t_1[:, 0] * vel_t_1[:, 1] - pos_t_1[:, 1] * vel_t_1[:, 0]) / (pos_t_1[:, 0] ** 2 + pos_t_1[:, 1] ** 2)
            hdg_t_1 = np.array(list(map(angle_to_pipi, np.arctan2(vel_t_1[:, 1], vel_t_1[:, 0]))))

            X = np.array([rad_t_1, np.cos(hdg_t_1), np.sin(hdg_t_1), vel_t_1[:, 0], vel_t_1[:, 1]])
            # Y = np.array([(rad_t-rad_t_1) / timestep, radial_vel_t])
            Y = np.array([vel_t[:, 0], vel_t[:, 1]])
            if inputs is None:
                inputs = X
                outputs = Y
            else:
                inputs = np.append(inputs, X, axis=1)
                outputs = np.append(outputs, Y, axis=1)
    return inputs, outputs


def split_data(data, timestep, split_func=split_cart, args={}):
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
        'files': files
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

    init_epoch = 0
    if args.load:
        model = tf.keras.models.load_model(Path(args.load), custom_objects={
            'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})

        ints = [int(s) for s in args.load.split('_') if s.isdigit()]
        init_epoch = ints[0]
    else:
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1],)))
        model.add(tf.keras.layers.Dense(50, activation='tanh'))
        model.add(tf.keras.layers.Dense(50, activation='tanh'))
        model.add(tf.keras.layers.Dense(Y.shape[1] * 2, activation=None))

        loss = gaussian_nll
        optimizer = tf.keras.optimizers.Adam(0.0001)
        model.compile(loss=loss,
                      optimizer=optimizer,
                      metrics=[gaussian_mse, gaussian_mae]
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
                'prob_' + str(epoch) + '_model.h5')))

    model.save(str(Path(args.path).joinpath('prob_model.h5')))
