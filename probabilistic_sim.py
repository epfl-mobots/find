#!/usr/bin/env python

import glob
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
import tensorflow.keras.backend as K

from features import Velocities
from utils import angle_to_pipi
from random import shuffle


from losses import *


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius < self._radius and radius > 0

    def center(self):
        return self._center


def sample_valid_velocity(ref_positions, generated_data, prediction, idx, setup):
    failed = 0
    while True:
        sample_velx = np.random.normal(
            prediction[0, 0], prediction[0, 2], 1)[0]
        sample_vely = np.random.normal(
            prediction[0, 1], prediction[0, 3], 1)[0]

        x_hat = generated_data[-2, idx * 2] + sample_velx * args.timestep
        y_hat = generated_data[-2, idx * 2 + 1] + sample_vely * args.timestep

        r = np.sqrt(
            (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)
            
        rv = np.sqrt(sample_velx ** 2 +
                sample_vely ** 2 +
                2 * sample_velx * sample_vely * np.cos(np.arctan2(sample_vely, sample_velx)))

        if setup.is_valid(r) and rv <= 1.2:
            generated_data[-1, idx * 2] = x_hat
            generated_data[-1, idx * 2 + 1] = y_hat
            sigmas.append(prediction[0, 2:])
            break
        else:
            # rold = np.sqrt((generated_data[-2, idx * 2] - setup.center()[0]) ** 2 + (
                # generated_data[-2, idx * 2 + 1] - setup.center()[1]) ** 2)

            failed += 1
            if failed > 999:
                prediction[:, 2] += 0.01
                prediction[:, 3] += 0.01
    return generated_data


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
                        required=False,
                        default=-1)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--exclude-index', '-e', type=int,
                        help='Index of the individual that will be replaced by a virtual agent (-1 will replace all original trajectories)',
                        required=False,
                        default=-2)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
                                       'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})
    setup = CircularCorridor()

    inputs = None
    outputs = None
    ref_positions = np.loadtxt(args.reference)
    ref_velocities = np.loadtxt(args.reference.replace('positions', 'velocities'))
    timestep = args.timestep

    pos_t = np.roll(ref_positions, shift=1, axis=0)[2:, :]
    pos_t_1 = np.roll(ref_positions, shift=1, axis=0)[1:-1, :]
    vel_t = np.roll(ref_velocities, shift=1, axis=0)[2:, :]
    vel_t_1 = np.roll(ref_velocities, shift=1, axis=0)[1:-1, :]

    assert args.exclude_index <  ref_positions.shape[1] // 2, 'Dimensions do not match'

    individuals = {}
    for idx, ind in enumerate(range(ref_positions.shape[1] // 2)):
        ix = ind * 2 
        iy = ind * 2 + 1

        individuals[idx] = {}
        individuals[idx]['pos'] = pos_t_1[:, (ind * 2) : (ind * 2 + 2)]
        individuals[idx]['vel'] = vel_t_1[:, (ind * 2) : (ind * 2 + 2)]

    # main simulation loop
    if args.iterations < 0:
        iters = ref_positions.shape[0]
    else:
        iters = args.iterations

    sigmas = []

    bootstrap = []
    for i in range(len(individuals.keys())):
        bootstrap.append(individuals[i]['pos'][0][0])
        bootstrap.append(individuals[i]['pos'][0][1])
    generated_data = np.matrix(bootstrap)
    
    if args.exclude_index >= 0:
        idcs = list(range(len(individuals.keys())))
        idcs.remove(args.exclude_index)
            
    for t in range(iters):
        if t % 500 == 0:
            print('Current timestep: ' + str(t))

        X = []
        if args.exclude_index >= 0:
            for i in range(ref_positions.shape[1] // 2):
                if args.exclude_index != i:
                    X.append(ref_positions[t, i * 2])
                    X.append(ref_positions[t, i * 2 + 1])
                    X.append(ref_velocities[t, i * 2])
                    X.append(ref_velocities[t, i * 2 + 1])
                else:
                    if t == 0:
                        X = [ref_positions[t, i * 2], ref_positions[t, i * 2 + 1], 
                             ref_velocities[t, i * 2], ref_velocities[t, i * 2 + 1]] + X
                    else:
                        x = generated_data[t, args.exclude_index * 2]
                        y = generated_data[t, args.exclude_index * 2 + 1]
                        x_t_1 = generated_data[t-1, args.exclude_index * 2]
                        y_t_1 = generated_data[t-1, args.exclude_index * 2 + 1]
                        vx = (x - x_t_1) / args.timestep
                        vy = (y - y_t_1) / args.timestep
                        X = [x, y, vx, vy] + X
            X = np.array(X)

            prediction = np.array(model.predict(X.reshape(1, X.shape[0])))            

            def logbound(val, max_logvar=0, min_logvar=-10):
                logsigma = max_logvar - \
                    np.log(np.exp(max_logvar - val) + 1)
                logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
                return logsigma
            
            prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
            prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

            generated_data = np.vstack([generated_data, ref_positions[t, :]]) 
            generated_data = sample_valid_velocity(ref_positions, generated_data, prediction, args.exclude_index, setup)
        else:
            ind_ids = list(range(ref_positions.shape[1] // 2))
            shuffle(ind_ids)

            for idx in ind_ids:
                X = []
                for i in range(ref_positions.shape[1] // 2):
                    if idx != i:
                        if ind_ids.index(idx) > ind_ids.index(i):
                            ts = t - 1
                        else:
                            ts = t
                        X.append(generated_data[ts, i * 2])
                        X.append(generated_data[ts, i * 2 + 1])
                        X.append((generated_data[ts, i * 2] - generated_data[ts - 1, i * 2]) / args.timestep)
                        X.append((generated_data[ts, i * 2 + 1] - generated_data[ts - 1, i * 2 + 1]) / args.timestep)
                    else:
                        if t == 0:
                            X = [ref_positions[t, i * 2], ref_positions[t, i * 2 + 1], 
                                ref_velocities[t, i * 2], ref_velocities[t, i * 2 + 1]] + X
                        else:
                            x = generated_data[t, idx * 2]
                            y = generated_data[t, idx * 2 + 1]
                            x_t_1 = generated_data[t-1, idx * 2]
                            y_t_1 = generated_data[t-1, idx * 2 + 1]
                            vx = (x - x_t_1) / args.timestep
                            vy = (y - y_t_1) / args.timestep
                            X = [x, y, vx, vy] + X
                X = np.array(X)

                prediction = np.array(model.predict(X.reshape(1, X.shape[0])))            

                def logbound(val, max_logvar=0, min_logvar=-10):
                    logsigma = max_logvar - \
                        np.log(np.exp(max_logvar - val) + 1)
                    logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
                    return logsigma
                
                prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
                prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

                if generated_data.shape[0] == t + 1:
                    generated_data = np.vstack([generated_data, ref_positions[t, :]]) 
                generated_data = sample_valid_velocity(ref_positions, generated_data, prediction, idx, setup)            

    gp_fname = args.reference.replace('processed', 'generated')
    sigma_fname = gp_fname.replace('positions', 'sigmas')
    gv_fname = gp_fname.replace('positions', 'velocities')
    gv = Velocities([np.array(generated_data)], args.timestep).get()

    np.savetxt(gp_fname, generated_data)
    np.savetxt(gv_fname, gv[0])
    np.savetxt(sigma_fname, np.array(sigmas))

