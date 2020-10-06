#!/usr/bin/env python

import tqdm
import argparse
from pathlib import Path
from random import shuffle

import tensorflow as tf

from utils.features import Velocities
from utils.losses import *

from simulation.fish_simulation import FishSimulation
from simulation.replay_individual import ReplayIndividual
from simulation.nn_individual import NNIndividual
from simulation.nn_functors import Multi_pfw_predict
from simulation.position_stat import PositionStat
from simulation.velocity_stat import VelocityStat


def cart_sim(model, args):
    inputs = None
    outputs = None
    ref_positions = np.loadtxt(args.reference)
    ref_velocities = np.loadtxt(
        args.reference.replace('positions', 'velocities'))
    timestep = args.timestep

    pos_t = np.roll(ref_positions, shift=1, axis=0)[2:, :]
    pos_t_1 = np.roll(ref_positions, shift=1, axis=0)[1:-1, :]
    vel_t = np.roll(ref_velocities, shift=1, axis=0)[2:, :]
    vel_t_1 = np.roll(ref_velocities, shift=1, axis=0)[1:-1, :]
    assert args.exclude_index < ref_positions.shape[1] // 2, 'Dimensions do not match'

    # initializing the simulation
    # if the trajectory is replayed then -1 makes sure that the last model prediction is not added to the generated file to ensure equal size of moves
    iters = pos_t_1.shape[0] - 1 if args.iterations < 0 else args.iterations
    simu_args = {'stats_enabled': True, 'simu_dir_gen': False}
    simu = FishSimulation(args.timestep, iters, args=simu_args)

    # setup predction object
    multi_pfw_interact = Multi_pfw_predict(model)

    # adding individuals to the simulation
    for i in range(pos_t_1.shape[1] // 2):
        if args.exclude_index > -1:
            if args.exclude_index == i:
                simu.add_individual(
                    NNIndividual(
                        multi_pfw_interact,
                        initial_pos=pos_t_1[0, (i * 2): (i * 2 + 2)],
                        initial_vel=vel_t_1[0, (i * 2): (i * 2 + 2)]))
            else:
                p = pos_t_1[:, (i * 2): (i * 2 + 2)]
                v = vel_t_1[:, (i * 2): (i * 2 + 2)]
                simu.add_individual(ReplayIndividual(p, v))
        else:  # purely virtual simulation
            simu.add_individual(
                NNIndividual(
                    multi_pfw_interact,
                    initial_pos=pos_t_1[0, (i * 2): (i * 2 + 2)],
                    initial_vel=vel_t_1[0, (i * 2): (i * 2 + 2)]))

    if args.exclude_index < 0:
        for _ in range(args.num_extra_virtu):
            simu.add_individual(
                NNIndividual(
                    multi_pfw_interact,
                    initial_pos=pos_t_1[0, 0:2],
                    initial_vel=vel_t_1[0, 0:2]))

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

    # execute the full simulation
    simu.spin()


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
                        default=-1)
    parser.add_argument('--polar', action='store_true',
                        help='Use polar inputs instead of cartesian coordinates',
                        default=False)
    parser.add_argument('--num-extra-virtu', type=int,
                        help='Number of virtual individuals in the simulation',
                        default=0)
    args = parser.parse_args()

    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
        'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})

    if not args.polar:
        cart_sim(model, args)
    else:
        # polar_sim(model, setup, args)
        assert False, 'This needs to be re-implemented for use with the particle simu'
