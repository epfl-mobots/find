#!/usr/bin/env python

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import matplotlib
from tqdm import tqdm

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
    parser.add_argument('--velocity-coef', type=float,
                        help='Scaling coefficient for the resultant velocity',
                        required=False,
                        default=1)
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
    vel = np.loadtxt(args.path.replace('positions', 'velocities'))
    if args.range is not None:  # keep the timesteps defined by the CLI parameters
        idcs = list(map(int, args.range))
        traj = traj[idcs[0]:idcs[1], :]
        vel = vel[idcs[0]:idcs[1], :]
    tsteps = traj.shape[0]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in tqdm(range(tsteps)):
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
                    # Q = plt.quiver(
                    #     x, y, vel[i, j * 2], vel[i, j * 2 + 1], scale=1, units='xy', color='r')

                    rvel = np.sqrt((vel[i, j * 2] * args.velocity_coef) ** 2 + (vel[i, j * 2 + 1] * args.velocity_coef)
                                   ** 2 - 2 * np.abs(vel[i, j * 2] * args.velocity_coef) * np.abs(vel[i, j * 2 + 1] * args.velocity_coef) * np.cos(np.arctan2(vel[i, j * 2 + 1], vel[i, j * 2])))
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
