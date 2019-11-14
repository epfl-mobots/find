#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


# def main():
#     x = np.linspace(0, 10, 20)
#     y = np.cos(x)
#     image_path = get_sample_data('ada.png')
#     fig, ax = plt.subplots()
#     imscatter(x, y, image_path, zoom=0.1, ax=ax)
#     ax.plot(x, y)
#     plt.show()


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
    args = parser.parse_args()

    iradius = 0.655172413793
    oradius = 1.0
    center = (0, 0)

    if args.dark:
        image_path = os.getcwd() + '/scripts/fish_dark.png'
    else:
        image_path = os.getcwd() + '/scripts/fish.png'
    image = Image.open(image_path)
    image_path = os.getcwd() + '/scripts/excluded.png'
    excluded_image = Image.open(image_path)
    image_path = os.getcwd() + '/scripts/excluded_t_1.png'
    excluded_image_t_1 = Image.open(image_path)
    image_path = os.getcwd() + '/scripts/robot.png'
    rimage = Image.open(image_path)

    traj = np.loadtxt(args.path)
    vel = np.loadtxt(args.path.replace('positions', 'velocities'))
    tsteps = traj.shape[0]

    rtraj = np.roll(traj, 1, axis=0)
    rvel = np.roll(vel, 1, axis=0)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(tsteps):
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
                phi = np.arctan2(vel[i, j * 2 + 1], vel[i, j * 2]) * 180 / np.pi
                if args.exclude_index == j:
                    rotated_img = excluded_image.rotate(phi)
                else:
                    rotated_img = image.rotate(phi)
                ax.imshow(rotated_img, extent=[x - 0.06, x + 0.06, y -
                                               0.06, y + 0.06], aspect='equal')

        ax.axis('off')

        if args.info:
            plt.legend(bbox_to_anchor=(0.93, 1.16),
                       bbox_transform=ax.transAxes)
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
