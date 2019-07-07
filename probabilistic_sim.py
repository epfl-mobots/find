#!/usr/bin/env python

import glob
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K

from features import Velocities
from utils import angle_to_pipi


from losses import *


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius < self._radius and radius > 0

    def center(self):
        return self._center


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Probabilistic model to reproduce fish motion')
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
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
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

    generated_data = np.matrix([X[0, 0], X[0, 1]])
    for t in range(args.iterations-1):
        print('Current timestep: ' + str(t))

        if t == 0:
            prediction = np.array(model.predict(X[0].reshape(1, X.shape[1])))
        else:
            dvel_t = (generated_data[-1, :] -
                      generated_data[-2, :]) / args.timestep
            nninput = np.array(
                [generated_data[-1, 0], generated_data[-1, 1], dvel_t[0, 0], dvel_t[0, 1]]).transpose()
            prediction = np.array(model.predict(
                nninput.reshape(1, X.shape[1])))

        max_logvar = -2
        min_logvar = -10
        logsigma = max_logvar - \
            np.log(np.exp(max_logvar - prediction[0, 2:]) + 1)
        logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)

        prediction[:, 2:] = list(map(np.exp, logsigma))
        # print(prediction[0, 2], prediction[0, 3])
        # input('')

        failed = 0
        while True:
            sample_velx = np.random.normal(
                prediction[0, 0], prediction[0, 2], 1)[0]
            sample_vely = np.random.normal(
                prediction[0, 1], prediction[0, 3], 1)[0]

            x_hat = generated_data[-1, 0] + sample_velx * args.timestep
            y_hat = generated_data[-1, 1] + sample_vely * args.timestep

            r = np.sqrt(
                (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)
            if setup.is_valid(r):
                generated_data = np.vstack([generated_data, [x_hat, y_hat]])
                break
            else:
                rold = np.sqrt((generated_data[-1, 0] - setup.center()[0]) ** 2 + (
                    generated_data[-1, 1] - setup.center()[1]) ** 2)

                failed += 1
                print(r, rold, prediction[0, 2], prediction[0, 3])
                # if failed > 999:
                #     # input('couldn not solve press any key')
                #     prediction[0, 0] = 0
                #     prediction[0, 1] = 0

    gp_fname = args.reference.replace('processed', 'generated')
    gv_fname = gp_fname.replace('positions', 'velocities')

    print(generated_data.shape)
    print(ref_positions[:10, :])
    print(generated_data[:10, :])
    gv = Velocities([np.array(generated_data)], args.timestep).get()

    np.savetxt(gp_fname, generated_data)
    np.savetxt(gv_fname, gv[0])

    #     controller = nn_models[prediction].predict(inputs[sample_idx, :].reshape(1, -1)).flatten()
    #     radius = cc.get_inner_radius() + controller[0]
    #     phi = to_minus180_180(controller[1] * 360) * np.pi / 180
    #     x, y = pol2cart(radius, phi, cc.get_center())
    #     controller[0] = x
    #     controller[1] = y
    #     controller_output.append(controller)
    # np.savetxt(Path(args.etho).joinpath('predictions_' + et + '_' + str(replicate) + '.dat'), controller_output)
