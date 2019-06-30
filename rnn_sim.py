#!/usr/bin/env python

import glob
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K

from features import Velocities
from utils import angle_to_pipi
from losses import gaussian_nll_tanh, gaussian_mae



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
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={'gaussian_nll_tanh': gaussian_nll_tanh, 'gaussian_mae': gaussian_mae})
    setup = CircularCorridor()

    inputs = None
    outputs = None
    ref_positions = np.loadtxt(args.reference)
    timestep = args.timestep
    for n in range(ref_positions.shape[1] // 2):
        rads = np.zeros([ref_positions.shape[0], 2])
        rads[:, 0] = ref_positions[:, n*2] - setup.center()[0]
        rads[:, 1] = ref_positions[:, n*2+1] - setup.center()[1]
        phis = np.arctan2(rads[:, 1], rads[:, 0])
        rads[:, 0] = rads[:, 0] ** 2
        rads[:, 1] = rads[:, 1] ** 2
        rads = np.sqrt(rads[:, 0] + rads[:, 1])

        drads_t = (rads - np.roll(rads, shift=1, axis=0))[2:] / timestep
        drads_t_1 = (rads - np.roll(rads, shift=1, axis=0))[1:-1] / timestep
        dphis_t = (np.array(list(map(lambda x: angle_to_pipi(x), phis - np.roll(phis, shift=1, axis=0))))[2:]) / timestep
        dphis_t_1 = (np.array(list(map(lambda x: angle_to_pipi(x), phis - np.roll(phis, shift=1, axis=0))))[1:-1]) / timestep

        rads_t = rads[2:]
        rads_t_1 = rads[1:-1]
        phis_t = phis[2:]
        phis_t_1 = phis[1:-1]

        X = np.array([rads_t_1, np.cos(phis_t_1), np.sin(phis_t_1), drads_t_1, np.cos(dphis_t_1), np.sin(dphis_t_1)])
        Y = np.array([drads_t, np.cos(dphis_t), np.sin(dphis_t)])
        if inputs is None:
            inputs = X
            outputs = Y
        else:
            inputs = np.append(inputs, X, axis=1)
            outputs = np.append(outputs, Y, axis=1)
    X = X.transpose()
    Y = Y.transpose()
    
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    phi = np.arctan(X[0, 0, 2] / X[0, 0, 1])
    generated_data = np.matrix([0.1, phi])

    for t in range(args.iterations-1):
        print('Current timestep: ' + str(t))

        failed = 0
        while True:
            if t == 0:
                prediction = np.array(model.predict(X[0].reshape(1, 1, X.shape[2])))
            else:
                drad_t = (generated_data[-1, 0] - generated_data[-2, 0]) / args.timestep
                dphi_t = (angle_to_pipi(generated_data[-1, 1] - generated_data[-2, 1])) / args.timestep
                nninput = np.array([generated_data[-1, 0], np.cos(generated_data[-1, 1]), np.sin(generated_data[-1, 1]), drad_t, np.cos(dphi_t), np.sin(dphi_t)]).transpose()
                prediction = np.array(model.predict(nninput.reshape(1, 1, X.shape[2])))

            prediction[:, 3:] = (prediction[:, 3:] + 1) / 2

            sample_rad = np.random.uniform(prediction[0, 0], prediction[0, 3], 1)[0]

            s = np.sin(np.random.uniform(prediction[0, 2], prediction[0, 5], 1)[0])
            c = np.cos(np.random.uniform(prediction[0, 1], prediction[0, 4], 1)[0])
            sample_phi = np.arctan2(s, c)
        
            rad_hat = np.abs(generated_data[-1, 0] + sample_rad * args.timestep)
            phi_hat = generated_data[-1, 1] + sample_phi * args.timestep

            # print(prediction[:, 3:])
            # input('')
            
            if failed == 0:
                generated_data = np.vstack([generated_data, [rad_hat, phi_hat]])
            
            if setup.is_valid(generated_data[-1, 0]):
                break
            else: 
                generated_data[-1, 0] = rad_hat
                # generated_data[-1, 1] = phi_hat
                failed += 1
                if failed > 399:
                    failed = 0
                    generated_data[-1, 0] = 0.99
                    break

    gp_fname = args.reference.replace('processed', 'generated')
    gv_fname = gp_fname.replace('positions', 'velocities')

    gp = []
    for i in range(generated_data.shape[0]):
        gp.append([generated_data[i, 0] * np.cos(generated_data[i, 1]), generated_data[i, 0] * np.sin(generated_data[i, 1])])
    
    gp = np.array(gp)
    print(ref_positions[:10, :])
    print(gp[:10, :])
    gv = Velocities([gp], args.timestep).get()

    np.savetxt(gp_fname, gp)
    np.savetxt(gv_fname, gv[0])

        
    #     controller = nn_models[prediction].predict(inputs[sample_idx, :].reshape(1, -1)).flatten()
    #     radius = cc.get_inner_radius() + controller[0]
    #     phi = to_minus180_180(controller[1] * 360) * np.pi / 180
    #     x, y = pol2cart(radius, phi, cc.get_center())
    #     controller[0] = x
    #     controller[1] = y
    #     controller_output.append(controller)
    # np.savetxt(Path(args.etho).joinpath('predictions_' + et + '_' + str(replicate) + '.dat'), controller_output)
