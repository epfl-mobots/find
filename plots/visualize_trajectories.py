#!/usr/bin/env python

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import matplotlib
from tqdm import tqdm
from utils.features import Velocities

matplotlib.use('Agg')


def get_text(fl):
    text = ''
    for pair in fl:
        if pair[0].lower() in ['alignment']:
            text += pair[0].capitalize() + ': ' + str('{0:.2f}'.format(
                np.asscalar(pair[1][i]) * 100)) + ' %\n'
        if pair[0].lower() in ['interindividual']:
            text += pair[0].capitalize() + ': ' + str('{0:.2f}'.format(
                np.asscalar(pair[1][i]) * 360)) + ' deg\n'
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions of the fish accompanied by the feature information')
    parser.add_argument('--path', '-p', type=str,
                        help='Positions path',
                        required=True)
    parser.add_argument('--out-dir', '-o', type=str,
                        help='Output directory name',
                        required=True)
    parser.add_argument('--fish-like', action='store_true',
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
    parser.add_argument('--exclude-index', '-e', type=int,
                        help='Index of the virtual individual',
                        required=False,
                        default=-1)
    parser.add_argument('--range', nargs='+',
                        help='Vector containing the start and end index of trajectories to be plotted',
                        required=False)
    parser.add_argument('--radius', '-r', type=float,
                        help='Raidus',
                        default=0.25,
                        required=False)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--dpi', type=int,
                        help='Raidus',
                        default=300,
                        required=False)
    parser.add_argument('--fill-between', type=int,
                        help='Fill frames between timesteps',
                        default=0,
                        required=False)
    args = parser.parse_args()

    iradius = 0.655172413793
    oradius = 1.0
    center = (0, 0)

    if args.dark:
        image_path = os.getcwd() + '/plots/fish_dark.png'
    else:
        image_path = os.getcwd() + '/plots/fish.png'
    image = Image.open(image_path)
    image_path = os.getcwd() + '/plots/excluded.png'
    excluded_image = Image.open(image_path)
    image_path = os.getcwd() + '/plots/excluded_t_1.png'
    excluded_image_t_1 = Image.open(image_path)
    image_path = os.getcwd() + '/plots/robot.png'
    rimage = Image.open(image_path)

    traj = np.loadtxt(args.path)
    vel = Velocities([traj * args.radius], args.timestep).get()[0]

    if args.range is not None:  # keep the timesteps defined by the CLI parameters
        idcs = list(map(int, args.range))
        traj = traj[idcs[0]:idcs[1], :]
        vel = vel[idcs[0]:idcs[1], :]
    tsteps = traj.shape[0]

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

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in tqdm(range(tsteps-1)):
        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()

        radius = (iradius, oradius)
        # plt.plot([center[0]], [center[1]], ls='none',
        #  marker='o', color='black', label='Origin ' + str(center))
        if args.dark:
            color = 'white'
        else:
            color = 'black'
        inner = plt.Circle(
            center, radius[0], color=color, fill=False)
        outer = plt.Circle(
            center, radius[1], color=color, fill=False)
        # ax.add_artist(inner)
        ax.add_artist(outer)

        for inum, j in enumerate(range(int(traj.shape[1] / 2))):
            x = traj[i, j * 2]
            y = traj[i, j * 2 + 1]

            if not args.fish_like:
                plt.scatter(x, y, marker='.',
                            label='Individual ' + str(inum) + ' ' + "{:.2f}".format(x) + ' ' + "{:.2f}".format(y))
                Q = plt.quiver(
                    x, y, vel[i, j * 2], vel[i, j * 2 + 1], scale=1, units='xy')
            else:
                phi = np.arctan2(vel[i, j * 2 + 1],
                                 vel[i, j * 2]) * 180 / np.pi
                if args.exclude_index == j:
                    rotated_img = excluded_image.rotate(phi)
                else:
                    rotated_img = image.rotate(phi)
                ax.imshow(rotated_img, extent=[x - 0.06, x + 0.06, y -
                                               0.06, y + 0.06], aspect='equal')

            if args.info:
                rvel = np.sqrt((vel[i, j * 2]) ** 2 + (vel[i, j * 2 + 1])
                               ** 2 - 2 * np.abs(vel[i, j * 2]) * np.abs(vel[i, j * 2 + 1]) * np.cos(np.arctan2(vel[i, j * 2 + 1], vel[i, j * 2])))
                plt.text(x + 0.025, y + 0.025,
                         "{:.4f}".format(rvel) + ' m/s', color='r', fontsize=5)

        ax.axis('off')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        plt.tight_layout()

        png_fname = args.out_dir + '/' + str(i).zfill(6)
        plt.savefig(
            str(png_fname) + '.png',
            transparent=True,
            dpi=300
        )
        plt.close('all')
