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

from simulation.fish_simulation import FishSimulation
from simulation.replay_individual import ReplayIndividual
from simulation.nn_individual import NNIndividual
from simulation.nn_functors import Multi_plstm_predict, Multi_plstm_predict_traj
from simulation.position_stat import PositionStat
from simulation.velocity_stat import VelocityStat
from simulation.nn_prediction_stat import NNPredictionStat


def cart_sim(model, args):
    ref_positions = np.loadtxt(args.reference)

    offset = 1
    if args.timesteps_skip > 0:
        offset = args.timesteps_skip

    pos_t_1 = np.roll(ref_positions, shift=1, axis=0)[
        1:-offset, :]
    pos_t = ref_positions[offset:-1, :]

    vel_t = (pos_t - pos_t_1) / args.timestep
    vel_t_1 = np.roll(vel_t, shift=1, axis=0)

    pos_t_1 = pos_t_1[1:-1, :]
    vel_t_1 = vel_t_1[1:-1, :]
    pos_t = pos_t[1:-1, :]
    vel_t = vel_t[1:-1, :]

    if args.timesteps_skip > 0:  # TODO: here we should run args.timesteps_skip simulations for more data
        pos_t_1 = pos_t_1[::(args.timesteps_skip + 1)]
        vel_t_1 = vel_t_1[::(args.timesteps_skip + 1)]
        pos_t = pos_t[::(args.timesteps_skip + 1)]
        vel_t = vel_t[::(args.timesteps_skip + 1)]

    assert args.exclude_index < ref_positions.shape[1] // 2, 'Dimensions do not match'

    # initializing the simulation
    # if the trajectory is replayed then -1 makes sure that the last model prediction is not added to the generated file to ensure equal size of moves
    iters = pos_t_1.shape[0] - \
        2 * args.num_timesteps if args.iterations < 0 else args.iterations
    simu_args = {'stats_enabled': True, 'simu_dir_gen': False}
    simu = FishSimulation(args.timestep, iters, args=simu_args)

    if args.prediction_steps > 1:
        multi_plstm_interact = Multi_plstm_predict_traj(
            model, args.num_timesteps)
    else:
        multi_plstm_interact = Multi_plstm_predict(model, args.num_timesteps)

    if iters - args.num_timesteps <= 0:
        import warnings
        warnings.warn('Skipping small simulation')
        exit(1)
    # adding individuals to the simulation
    for i in range(pos_t_1.shape[1] // 2):
        if args.exclude_index > -1:
            if args.exclude_index == i:
                simu.add_individual(
                    NNIndividual(
                        multi_plstm_interact,
                        initial_pos=pos_t_1[:args.num_timesteps,
                                            (i * 2): (i * 2 + 2)],
                        initial_vel=vel_t_1[:args.num_timesteps, (i * 2): (i * 2 + 2)]))
            else:
                p = pos_t_1[args.num_timesteps:, (i * 2): (i * 2 + 2)]
                v = vel_t_1[args.num_timesteps:, (i * 2): (i * 2 + 2)]
                simu.add_individual(ReplayIndividual(p, v))
        else:  # purely virtual simulation
            simu.add_individual(
                NNIndividual(
                    multi_plstm_interact,
                    initial_pos=pos_t_1[:args.num_timesteps,
                                        (i * 2): (i * 2 + 2)],
                    initial_vel=vel_t_1[:args.num_timesteps, (i * 2): (i * 2 + 2)]))

    # generated files have different names if the simulation is virtual or hybrid
    if args.exclude_index < 0:
        gp_fname = args.reference.replace('processed', 'generated_virtu')
    else:
        gp_fname = args.reference.replace(
            'processed', 'idx_' + str(args.exclude_index) + '_generated')
    gv_fname = gp_fname.replace('positions', 'velocities')

    # adding stat objects
    simu.add_stat(PositionStat(pos_t_1.shape[1], gp_fname))
    simu.add_stat(VelocityStat(vel_t_1.shape[1], gv_fname))
    if args.exclude_index > -1 and (pos_t.shape[1] // 2) == 1:
        ind_idcs = list(range(pos_t.shape[1] // 2))
        ind_idcs.remove(args.exclude_index)
        simu.add_stat(NNPredictionStat(multi_plstm_interact,
                                       pos_t_1.shape[1],
                                       args.reference.replace('processed',
                                                              'idx_' + str(ind_idcs[0]) + '_predicted')))

    # execute the full simulation
    simu.spin()


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
    parser.add_argument('--polar', action='store_true',
                        help='Use polar inputs instead of cartesian coordinates',
                        default=False)
    parser.add_argument('--timesteps-skip', type=int,
                        help='Timesteps skipped between input and prediction',
                        default=0,
                        required=False)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
        'Y': np.empty((0, 2)),
        'multi_dim_gaussian_nll': multi_dim_gaussian_nll,
        'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})

    if not args.polar:
        cart_sim(model, args)
    else:
        # polar_sim(model, setup, args)
        assert False, 'This needs to be re-implemented for use with the particle simu'
