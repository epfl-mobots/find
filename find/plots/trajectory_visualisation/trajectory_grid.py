#!/usr/bin/env python

import os
import glob
import random
import argparse
import numpy as np

from find.utils.features import Velocities
from find.plots.common import *


def plot(foo, path, args):
    trajectories = []
    if args.traj_visualisation_list == 'random':
        files = glob.glob(args.path + '/generated/*_positions.dat')
        trajectories.append(np.loadtxt(random.choice(files)) * args.radius)
    else:
        files = glob.glob(args.traj_visualisation_list)
        for f in files:
            trajectories.appened(np.loadtxt(f) * args.radius)

    for fidx, traj in enumerate(trajectories):
        vel = Velocities([traj], args.timestep).get()[0]
        lb, ub = 0, traj.shape[0]
        if args.range:
            lb, ub = args.range

        rvelocities = []
        for ind in range(traj.shape[1] // 2):
            for i in range(traj.shape[0]):
                r = np.sqrt(vel[i, ind * 2 + 1] ** 2 +
                            vel[i, ind * 2] ** 2 -
                            2 * np.abs(vel[i, ind * 2 + 1]) * np.abs(vel[i, ind * 2]) * np.cos(
                    np.arctan2(vel[i, ind * 2 + 1], vel[i, ind * 2])))
                rvelocities.append(r)
            _ = plt.figure(figsize=(5, 5))
            ax = plt.gca()
            outer = plt.Circle(
                args.center, args.radius, color='white', fill=False)
            ax.add_artist(outer)
            plt.plot(traj[lb:ub, ind * 2],
                     traj[lb:ub, ind * 2 + 1], linewidth=0.2)
        ax.set_xlim([-args.radius*1.05, args.radius*1.05])
        ax.set_ylim([-args.radius*1.05, args.radius*1.05])
        plt.savefig(path + os.path.basename(files[fidx]) + '.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions')
    parser.add_argument('--traj_visualisation_list',
                        type=str,
                        nargs='+',
                        help='List of files to visualise',
                        default='random',
                        required=False)
    parser.add_argument('--exclude_index', '-e', type=int,
                        help='Index of the virtual individual',
                        required=False,
                        default=-1)
    parser.add_argument('--timesteps', '-t', type=int,
                        default=-1,
                        help='Timesteps to use in the plot',
                        required=False)
    parser.add_argument('--range', nargs='+',
                        help='Vector containing the start and end index of trajectories to be plotted',
                        required=False)
    parser.add_argument('--radius', '-r', type=float,
                        help='Radius',
                        default=0.25,
                        required=False)
    parser.add_argument('--center',
                        type=float,
                        nargs='+',
                        help='The centroidal coordinates for the setups used',
                        default=[0.0, 0.0],
                        required=False)
    args = parser.parse_args()

    plot(None, './', args)
