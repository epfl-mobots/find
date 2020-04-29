#!/usr/bin/env python

import tqdm
import argparse
from pathlib import Path

import tensorflow as tf

from utils.features import Velocities
from utils.losses import *


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
        'Y': np.empty((0, 2))})
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

    sigmas = []
    for t in tqdm.tqdm(range((iterations - 1) // args.prediction_steps)):
        if t == 0:
            prediction = np.array(model.predict(
                X[:args.num_timesteps].reshape(1, args.num_timesteps, X.shape[2])))
        else:
            dvel_t = (generated_pos[-1, :] -
                      generated_pos[-2, :]) / args.timestep

            previous_tstep = np.array(
                [generated_pos[-1, 0], generated_pos[-1, 1], dvel_t[0, 0], dvel_t[0, 1]]).T

            nninput = np.hstack(
                [generated_pos[(-args.num_timesteps+1):, :], generated_vel[(-args.num_timesteps+1):, :]])
            nninput = np.array(np.vstack([nninput, previous_tstep]))
            nninput = np.reshape(
                nninput, (1, args.num_timesteps, X.shape[2]))
            prediction = model.predict(nninput)

            if args.prediction_steps > 1:
                for k in range(args.prediction_steps):
                    failed = 0
                    noise = 0.0
                    while True:
                        sample_velx = np.random.normal(
                            prediction[0, 0, k, 0], noise, 1)[0]
                        sample_vely = np.random.normal(
                            prediction[0, 0, k, 1], noise, 1)[0]

                        x_hat = generated_pos[-1, 0] + \
                            sample_velx * args.timestep
                        y_hat = generated_pos[-1, 1] + \
                            sample_vely * args.timestep

                        r = np.sqrt(
                            (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)
                        if setup.is_valid(r):
                            generated_pos = np.vstack(
                                [generated_pos, [x_hat, y_hat]])
                            dvel_t = (generated_pos[-1, :] -
                                      generated_pos[-2, :]) / args.timestep
                            generated_vel = np.vstack(
                                [generated_vel, [dvel_t[0, 0], dvel_t[0, 1]]])
                            break
                        else:
                            rold = np.sqrt((generated_pos[-1, 0] - setup.center()[0]) ** 2 + (
                                generated_pos[-1, 1] - setup.center()[1]) ** 2)
                            failed += 1
                            if failed > 999:
                                noise += 0.01
            else:
                failed = 0
                noise = 0.0
                while True:
                    sample_velx = np.random.normal(
                        prediction[0, 0], noise, 1)[0]
                    sample_vely = np.random.normal(
                        prediction[0, 1], noise, 1)[0]

                    x_hat = generated_pos[-1, 0] + \
                        sample_velx * args.timestep
                    y_hat = generated_pos[-1, 1] + \
                        sample_vely * args.timestep

                    r = np.sqrt(
                        (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)
                    if setup.is_valid(r):
                        generated_pos = np.vstack(
                            [generated_pos, [x_hat, y_hat]])
                        dvel_t = (generated_pos[-1, :] -
                                  generated_pos[-2, :]) / args.timestep
                        generated_vel = np.vstack(
                            [generated_vel, [dvel_t[0, 0], dvel_t[0, 1]]])
                        break
                    else:
                        rold = np.sqrt((generated_pos[-1, 0] - setup.center()[0]) ** 2 + (
                            generated_pos[-1, 1] - setup.center()[1]) ** 2)
                        failed += 1
                        if failed > 999:
                            noise += 0.01

        gp_fname = args.reference.replace('processed', 'generated')
        gv_fname = gp_fname.replace('positions', 'velocities')
        gv = Velocities([np.array(generated_pos)], args.timestep).get()

        np.savetxt(gp_fname, generated_pos)
        np.savetxt(gv_fname, gv[0])
