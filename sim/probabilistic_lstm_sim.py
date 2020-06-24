#!/usr/bin/env python

import tqdm
import argparse
from pathlib import Path

import tensorflow as tf

from utils.features import Velocities
from utils.losses import *

from models.probabilistic_lstm_model import multi_dim_gaussian_nll


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius <= self._radius

    def center(self):
        return self._center


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RNN model to reproduce fish motion')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--reference', '-r', type=str,
                        help='Path to a reference experiment position file',
                        required=True)
    parser.add_argument('--model', '-m', type=str,
                        help='Model file name to use',
                        required=True)
    parser.add_argument('--iterations', '-i', type=int,
                        help='Number of iteration of the simulation',
                        default=-1,
                        required=False)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--num-timesteps', type=int,
                        help='Number of LSTM timesteps',
                        default=5)
    parser.add_argument('--prediction-steps', type=int,
                        help='Number of prediction steps for the NN',
                        default=1)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
        'Y': np.empty((0, 2)),
        'multi_dim_gaussian_nll': multi_dim_gaussian_nll,
        'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})
    setup = CircularCorridor()

    inputs = None
    outputs = None
    ref_positions = np.loadtxt(args.reference)
    timestep = args.timestep
    for n in range(ref_positions.shape[1] // 2):
        pos_t = np.roll(ref_positions, shift=1, axis=0)[2:, :]
        pos_t_1 = np.roll(ref_positions, shift=1, axis=0)[1:-1, :]
        vel_t = (ref_positions - np.roll(ref_positions,
                                         shift=1, axis=0))[2:, :] / timestep
        vel_t_1 = (ref_positions - np.roll(ref_positions,
                                           shift=1, axis=0))[1:-1, :] / timestep

        X = np.array([pos_t_1[:, 0], pos_t_1[:, 1],
                      vel_t_1[:, 0], vel_t_1[:, 1]])
        Y = np.array([vel_t[:, 0], vel_t[:, 1]])
        if inputs is None:
            inputs = X
            outputs = Y
        else:
            inputs = np.append(inputs, X, axis=1)
            outputs = np.append(outputs, Y, axis=1)
    X = X.transpose()
    Y = Y.transpose()

    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    # we start at the second frame of the original data to get the derivative for the velocity
    # for the requested timesteps
    generated_pos = np.matrix(
        [X[:args.num_timesteps, 0, 0], X[:args.num_timesteps, 0, 1]]).T

    generated_vel = np.matrix(
        [X[:args.num_timesteps, 0, 2], X[:args.num_timesteps, 0, 3]]).T

    iterations = args.iterations
    if iterations < 0:
        iterations = X.shape[0]

    def logbound(val, max_logvar=0, min_logvar=-10):
        logsigma = max_logvar - \
            np.log(np.exp(max_logvar - val) + 1)
        logsigma = min_logvar + \
            np.log(np.exp(logsigma - min_logvar) + 1)
        return logsigma

    sigmas = []
    for t in tqdm.tqdm(range((iterations - 1) // args.prediction_steps)):
        if t == 0:
            prediction = np.array(model.predict(
                X[:args.num_timesteps].reshape(1, args.num_timesteps, X.shape[2])))
        else:
            nninput = np.hstack(
                [generated_pos[-args.num_timesteps:, :], generated_vel[-args.num_timesteps:, :]])
            nninput = np.array(nninput)
            nninput = np.reshape(
                nninput, (1, args.num_timesteps, X.shape[2]))
            prediction = model.predict(nninput)

        if args.prediction_steps > 1:
            for k in range(args.prediction_steps):
                prediction[0, 0, k, 2:] = list(
                    map(logbound, prediction[0, 0, k, 2:]))
                prediction[0, 0, k, 2:] = list(
                    map(np.exp, prediction[0, 0, k, 2:]))

                failed = 0
                while True:
                    sample_velx = np.random.normal(
                        prediction[0, 0, k, 0], prediction[0, 0, k, 2], 1)[0]
                    sample_vely = np.random.normal(
                        prediction[0, 0, k, 1], prediction[0, 0, k, 3], 1)[0]

                    x_hat = generated_pos[-1, 0] + \
                        sample_velx * args.timestep
                    y_hat = generated_pos[-1, 1] + \
                        sample_vely * args.timestep

                    r = np.sqrt(
                        (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)

                    rv = np.sqrt(sample_velx ** 2 +
                                    sample_vely ** 2 -
                                    2 * np.abs(sample_velx) * np.abs(sample_vely) * np.cos(np.arctan2(sample_vely, sample_velx)))

                    if setup.is_valid(r) and rv <= 1.2:
                        generated_pos = np.vstack(
                            [generated_pos, [x_hat, y_hat]])
                        dvel_t = (generated_pos[-1, :] -
                                    generated_pos[-2, :]) / args.timestep
                        generated_vel = np.vstack(
                            [generated_vel, [dvel_t[0, 0], dvel_t[0, 1]]])
                        sigmas.append(prediction[0, 0, k, 2:])
                        break
                    else:
                        rold = np.sqrt((generated_pos[-1, 0] - setup.center()[0]) ** 2 + (
                            generated_pos[-1, 1] - setup.center()[1]) ** 2)

                        failed += 1
                        if failed > 999:
                            prediction[0, 0, k, 2] += 0.01
                            prediction[0, 0, k, 3] += 0.01
        else:
            prediction[0, 2:] = list(
                map(logbound, prediction[0, 2:]))
            prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

            failed = 0
            while True:
                sample_velx = np.random.normal(
                    prediction[0, 0], prediction[0, 2], 1)[0]
                sample_vely = np.random.normal(
                    prediction[0, 1], prediction[0, 3], 1)[0]

                x_hat = generated_pos[-1, 0] + sample_velx * args.timestep
                y_hat = generated_pos[-1, 1] + sample_vely * args.timestep

                r = np.sqrt(
                    (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)

                rv = np.sqrt(sample_velx ** 2 +
                                sample_vely ** 2 -
                                2 * np.abs(sample_velx) * np.abs(sample_vely) * np.cos(np.arctan2(sample_vely, sample_velx)))

                if setup.is_valid(r) and rv <= 1.2:
                    generated_pos = np.vstack(
                        [generated_pos, [x_hat, y_hat]])
                    dvel_t = (generated_pos[-1, :] -
                                generated_pos[-2, :]) / args.timestep
                    generated_vel = np.vstack(
                        [generated_vel, [dvel_t[0, 0], dvel_t[0, 1]]])
                    sigmas.append(prediction[0, 2:])
                    break
                else:
                    rold = np.sqrt((generated_pos[-1, 0] - setup.center()[0]) ** 2 + (
                        generated_pos[-1, 1] - setup.center()[1]) ** 2)

                    failed += 1
                    if failed > 999:
                        prediction[0, 2] += 0.01
                        prediction[0, 3] += 0.01

    gp_fname = args.reference.replace('processed', 'generated')
    gv_fname = gp_fname.replace('positions', 'velocities')
    gv = Velocities([np.array(generated_pos)], args.timestep).get()

    np.savetxt(gp_fname, generated_pos)
    np.savetxt(gv_fname, gv[0])
