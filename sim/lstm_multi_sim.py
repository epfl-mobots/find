#!/usr/bin/env python

import tqdm
import argparse
import glob
from pathlib import Path
from random import shuffle

import tensorflow as tf

from utils.features import Velocities
from utils.losses import *
from models.probabilistic_lstm_multi_model import multi_dim_gaussian_nll


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius <= self._radius

    def center(self):
        return self._center

def sample_valid_velocity(ref_positions, generated_data, prediction, idx, setup):
    failed = 0
    noise = 0.0
    while True:
        sample_velx = np.random.normal(
            prediction[0, 0], noise, 1)[0]
        sample_vely = np.random.normal(
            prediction[0, 1], noise, 1)[0]

        x_hat = generated_data[-2, idx * 2] + sample_velx * args.timestep
        y_hat = generated_data[-2, idx * 2 + 1] + sample_vely * args.timestep

        r = np.sqrt(
            (x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)

        rv = np.sqrt(sample_velx ** 2 +
                     sample_vely ** 2 -
                     2 * sample_velx * sample_vely * np.cos(np.arctan2(sample_vely, sample_velx)))

        if setup.is_valid(r) and rv <= 1.2:
            generated_data[-1, idx * 2] = x_hat
            generated_data[-1, idx * 2 + 1] = y_hat
            break
        else:
            failed += 1
            if failed > 999:
                noise += 0.01
    return generated_data




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
    parser.add_argument('--exclude-index', '-e', type=int,
                        help='Index of the individual that will be replaced by a virtual agent (-1 will replace all original trajectories)',
                        required=False,
                        default=-1)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
        'Y': np.empty((0, 2)),
        'multi_dim_gaussian_nll': multi_dim_gaussian_nll,
        'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})
    setup = CircularCorridor()

    inputs = None
    outputs = None
    ref_positions = np.loadtxt(args.reference)
    ref_velocities = np.loadtxt(
        args.reference.replace('positions', 'velocities'))
    timestep = args.timestep

    pos_t = np.roll(ref_positions, shift=1, axis=0)[2:, :]
    pos_t_1 = np.roll(ref_positions, shift=1, axis=0)[1:-1, :]
    vel_t = (ref_positions - np.roll(ref_positions,
                                        shift=1, axis=0))[2:, :] / timestep
    vel_t_1 = (ref_positions - np.roll(ref_positions,
                                        shift=1, axis=0))[1:-1, :] / timestep

    assert args.exclude_index < ref_positions.shape[1] // 2, 'Dimensions do not match'

    individuals = {}
    for idx, ind in enumerate(range(ref_positions.shape[1] // 2)):
        ix = ind * 2
        iy = ind * 2 + 1

        individuals[idx] = {}
        individuals[idx]['pos'] = pos_t_1[:, (ind * 2): (ind * 2 + 2)]
        individuals[idx]['vel'] = vel_t_1[:, (ind * 2): (ind * 2 + 2)]

    # main simulation loop
    if args.iterations < 0:
        iters = ref_positions.shape[0]
    else:
        iters = args.iterations

    # we are starting the generated data by copying args.num_timesteps rows 
    # to have sufficient input data for the lstm structure
    bootstrap = []
    for ts in range(args.num_timesteps+1):
        current_ts = []
        for i in range(len(individuals.keys())):
            current_ts.append(individuals[i]['pos'][ts][0])
            current_ts.append(individuals[i]['pos'][ts][1])
        bootstrap.append(current_ts)
    generated_data = np.array(bootstrap)

    if args.exclude_index >= 0:
        idcs = list(range(len(individuals.keys())))
        idcs.remove(args.exclude_index)

    # TODO: This does not include the simulation for multiple prediction steps
    for t in tqdm.tqdm(range(args.num_timesteps+1, (iters - 1) // args.prediction_steps)):
        if args.exclude_index >= 0:
            X = np.empty((args.num_timesteps, 0))

            for i in range(ref_positions.shape[1] // 2):
                if args.exclude_index != i:
                    X = np.hstack((X, ref_positions[(t-args.num_timesteps):t, (i * 2):(i * 2 + 2) ]))
                    X = np.hstack((X, ref_velocities[(t-args.num_timesteps):t, (i * 2):(i * 2 + 2) ]))
                else:
                    if t == args.num_timesteps + 1:
                        focal = np.empty((args.num_timesteps, 0))
                        focal = np.hstack((focal, ref_positions[(t-args.num_timesteps):t, (i * 2):(i * 2 + 2) ]))
                        focal = np.hstack((focal, ref_velocities[(t-args.num_timesteps):t, (i * 2):(i * 2 + 2) ]))
                        X = np.hstack((focal, X))
                    else:
                        xy = generated_data[(t-args.num_timesteps):t, (args.exclude_index * 2):(args.exclude_index * 2 + 2)]
                        xy_t_1 = generated_data[(t-args.num_timesteps-1):(t - 1), (args.exclude_index * 2):(args.exclude_index * 2 + 2)]
                        vxy = (xy - xy_t_1) / args.timestep

                        focal = np.empty((args.num_timesteps, 0))
                        focal = np.hstack((focal, xy))
                        focal = np.hstack((focal, vxy))
                        X = np.hstack((focal, X))

            prediction = np.array(model.predict(X.reshape(1, args.num_timesteps, X.shape[1])))

            # The following line might initially seem weird. What I do here 
            # is the following: I stack the reference positions vertically
            # and then when I call sample_valid_velocity I will in fact 
            # replace the reference positions that should correspond to the 
            # virtual agent with what was generated.
            generated_data = np.vstack([generated_data, ref_positions[t, :]])
            generated_data = sample_valid_velocity(
                ref_positions, generated_data, prediction, args.exclude_index, setup)
        else:
            ind_ids = list(range(ref_positions.shape[1] // 2))
            shuffle(ind_ids)

            for idx in ind_ids:
                X = np.empty((args.num_timesteps, 0))

                for i in range(ref_positions.shape[1] // 2):
                    if idx != i:
                        if ind_ids.index(idx) > ind_ids.index(i):
                            ts = t - 1
                        else:
                            ts = t
                        
                        X = np.hstack((X, ref_positions[(ts-args.num_timesteps):ts, (i * 2):(i * 2 + 2) ]))
                        X = np.hstack((X, ref_velocities[(ts-args.num_timesteps):ts, (i * 2):(i * 2 + 2) ]))
                    else:
                        if t == args.num_timesteps + 1:
                            focal = np.empty((args.num_timesteps, 0))
                            focal = np.hstack((focal, ref_positions[(t-args.num_timesteps):t, (i * 2):(i * 2 + 2) ]))
                            focal = np.hstack((focal, ref_velocities[(t-args.num_timesteps):t, (i * 2):(i * 2 + 2) ]))
                            X = np.hstack((focal, X))
                        else:
                            xy = generated_data[(t-args.num_timesteps):t, (idx * 2):(idx * 2 + 2)]
                            xy_t_1 = generated_data[(t-args.num_timesteps-1):(t - 1), (idx * 2):(idx * 2 + 2)]
                            vxy = (xy - xy_t_1) / args.timestep

                            focal = np.empty((args.num_timesteps, 0))
                            focal = np.hstack((focal, xy))
                            focal = np.hstack((focal, vxy))
                            X = np.hstack((focal, X))

                prediction = np.array(model.predict(X.reshape(1, args.num_timesteps, X.shape[1])))

                generated_data = np.vstack(
                        [generated_data, ref_positions[t, :]])
                generated_data = sample_valid_velocity(
                    ref_positions, generated_data, prediction, idx, setup)


    if args.exclude_index < 0:
        gp_fname = args.reference.replace('processed', 'generated_virtu')
    else:
        gp_fname = args.reference.replace('processed', 'idx_' + str(args.exclude_index) + '_generated')
    sigma_fname = gp_fname.replace('positions', 'sigmas')
    gv_fname = gp_fname.replace('positions', 'velocities')
    gv = Velocities([np.array(generated_data)], args.timestep).get()

    np.savetxt(gp_fname, generated_data)
    np.savetxt(gv_fname, gv[0])
    