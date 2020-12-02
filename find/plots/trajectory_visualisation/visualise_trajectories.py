#!/usr/bin/env python
import os
import glob
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import find.plots as fp
from find.utils.features import Velocities
from find.utils.utils import compute_leadership
from find.plots.common import *


def plot(foo, path, args):
    mpath = os.path.dirname(fp.__file__)

    # random select an experiment to visualise
    trajectories = []
    if args.traj_visualisation_list == 'random':
        files = glob.glob(args.path + '/generated/*_positions.dat')
        trajectories.append(np.loadtxt(random.choice(files)) * args.radius)
    else:
        files = glob.glob(args.traj_visualisation_list)
        for f in files:
            trajectories.appened(np.loadtxt(f) * args.radius)

    # TODO: parallelise multiple visualisations ?
    for fidx, traj in enumerate(trajectories):
        vel = Velocities([traj * args.radius], args.timestep).get()[0]

        if args.fish_like:
            pictures = []
            for i in range(traj.shape[1] // 2):
                pictures.append(Image.open(
                    mpath + '/res/fish{}.png'.format(i))
                    # .resize((45, 45))
                )

        # pick the range of trajectories to visualise
        if args.range is not None:  # keep the timesteps defined by the CLI parameters
            idcs = list(map(int, args.range))
            traj = traj[idcs[0]:idcs[1], :]
            vel = vel[idcs[0]:idcs[1], :]

        # In case the user wants to produce smoother videos (s)he can opt to fill frames between actual data points
        if args.fill_between > 0:
            filled_traj = np.empty(
                ((traj.shape[0] - 1) * args.fill_between, 0))
            filled_vel = np.empty(((traj.shape[0] - 1) * args.fill_between, 0))

            for idx in range(traj.shape[1] // 2):
                ft = np.empty((0, 2))
                fv = np.empty((0, 2))
                for i in tqdm(range(traj.shape[0] - 1), desc='filling trajectories'):
                    fill_x = np.linspace(
                        traj[i, idx * 2], traj[i + 1, idx * 2], args.fill_between)
                    fill_y = np.linspace(
                        traj[i, idx * 2 + 1], traj[i + 1, idx * 2 + 1], args.fill_between)
                    fill_vx = np.linspace(
                        vel[i, idx * 2], vel[i + 1, idx * 2], args.fill_between)
                    fill_vy = np.linspace(
                        vel[i, idx * 2 + 1], vel[i + 1, idx * 2 + 1], args.fill_between)
                    ft = np.vstack(
                        (ft, np.vstack((fill_x, fill_y)).T))
                    fv = np.vstack(
                        (fv, np.vstack((fill_vx, fill_vy)).T))
                filled_traj = np.hstack((filled_traj, ft))
                filled_vel = np.hstack((filled_vel, fv))
            traj = np.vstack((filled_traj, traj[-1, :]))
            vel = np.vstack((filled_vel, vel[-1, :]))

        if args.info:
            _, leadership_mat = compute_leadership(traj, vel)
            leadership_mat = np.array(leadership_mat)

        out_dir = path + '/' + os.path.basename(files[fidx]).split('.')[0]

        tsteps = traj.shape[0]
        for i in tqdm(range(tsteps-1)):
            _ = plt.figure(figsize=(5, 5))
            ax = plt.gca()

            if args.dark:
                color = 'white'
            else:
                color = 'black'
            outer = plt.Circle(
                args.center, args.radius, color=color, fill=False)
            ax.add_artist(outer)

            for inum, j in enumerate(range(traj.shape[1] // 2)):
                x = traj[i, j * 2]
                y = traj[i, j * 2 + 1]

                if not args.fish_like:
                    plt.scatter(x, y, marker='.',
                                label='Individual ' + str(inum) + ' ' + "{:.2f}".format(x) + ' ' + "{:.2f}".format(y))
                    plt.quiver(
                        x, y, vel[i, j * 2], vel[i, j * 2 + 1], scale=1, units='xy')
                else:
                    phi = np.arctan2(vel[i, j * 2 + 1],
                                     vel[i, j * 2]) * 180 / np.pi
                    rimage = pictures[j].rotate(phi)
                    ax.imshow(rimage, extent=[x - 0.018, x + 0.018, y -
                                              0.018, y + 0.018], aspect='equal')

                if args.info:
                    flag = leadership_mat[i, 1] == j
                    if flag:
                        fish_status = 'L'
                    else:
                        fish_status = 'F'
                    rvel = np.sqrt((vel[i, j * 2]) ** 2 + (vel[i, j * 2 + 1])
                                   ** 2 - 2 * np.abs(vel[i, j * 2]) * np.abs(vel[i, j * 2 + 1]) * np.cos(np.arctan2(vel[i, j * 2 + 1], vel[i, j * 2])))
                    plt.text(x + 0.01, y + 0.01,
                             fish_status + ': {:.4f}'.format(rvel) + ' m/s', color='r', fontsize=5)

            ax.axis('off')
            ax.set_xlim([-args.radius*1.05, args.radius*1.05])
            ax.set_ylim([-args.radius*1.05, args.radius*1.05])
            plt.tight_layout()

            png_fname = out_dir + '/' + str(i).zfill(6)
            plt.savefig(
                str(png_fname) + '.png',
                transparent=True,
                dpi=300
            )
            plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise the positions of the fish accompanied by the feature information')
    parser.add_argument('--traj_visualisation_list',
                        type=str,
                        nargs='+',
                        help='List of files to visualise',
                        default='random',
                        required=False)
    parser.add_argument('--fish_like', action='store_true',
                        help='Images instead of points',
                        default=False)
    parser.add_argument('--turing', action='store_true',
                        help='Same image for all individuals to perform a turing test',
                        default=False)
    parser.add_argument('--info', action='store_true',
                        help='Display info',
                        default=False)
    parser.add_argument('--dark', action='store_true',
                        help='Render dark friendly icons',
                        default=False)
    parser.add_argument('--exclude_index', '-e', type=int,
                        help='Index of the virtual individual',
                        required=False,
                        default=-1)
    parser.add_argument('--range', nargs='+',
                        help='Vector containing the start and end index of trajectories to be plotted',
                        required=False)
    parser.add_argument('--radius', '-r', type=float,
                        help='Radius',
                        default=0.25,
                        required=False)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--fill_between', type=int,
                        help='Fill frames between timesteps',
                        default=0,
                        required=False)
    parser.add_argument('--center',
                        type=float,
                        nargs='+',
                        help='The centroidal coordinates for the setups used',
                        default=[0.0, 0.0],
                        required=False)
    args = parser.parse_args()

    plot(None, './', args)
