#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')

import os
import glob
import tqdm
import argparse
import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

import tensorflow as tf

from utils.losses import *

plt.style.use('dark_background')

iradius = 0.655172413793
oradius = 1.0
center = (0, 0)
radius = (iradius, oradius)

# if args.dark:
image_path = os.getcwd() + '/plots/fish_dark.png'
# else:
#     image_path = os.getcwd() + '/plots/fish.png'
image = Image.open(image_path)
image_path = os.getcwd() + '/plots/excluded.png'
excluded_image = Image.open(image_path)
image_path = os.getcwd() + '/plots/excluded_t_1.png'
excluded_image_t_1 = Image.open(image_path)
image_path = os.getcwd() + '/plots/robot.png'
rimage = Image.open(image_path)


def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius < self._radius and radius > 0

    def center(self):
        return self._center


def cart_sim(model, setup, args):
    global radius, center

    inputs = None
    outputs = None
    p = np.loadtxt(args.reference)
    v = np.loadtxt(args.reference.replace('positions', 'velocities'))
    assert p.shape == v.shape, 'Dimensions don\'t match'

    pos_t = np.roll(p, shift=1, axis=0)[2:, :]
    pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
    vel_t = np.roll(v, shift=1, axis=0)[2:, :]
    vel_t_1 = np.roll(v, shift=1, axis=0)[1:-1, :]

    if args.iterations < 0:
        iters = p.shape[0]
    else:
        iters = args.iterations


    bins = 1000
    m_y, m_x = np.meshgrid(np.linspace(center[0] - (oradius + 0.0001),
                                   center[0] + (oradius + 0.0001), bins),
                       np.linspace(center[1] - (oradius + 0.0001),
                                   center[1] + (oradius + 0.0001), bins))
    r = np.sqrt((m_x - center[0]) ** 2 + (m_y - center[1]) ** 2)
    outside_els = np.sum(r > radius[1])


    for t in tqdm.tqdm(range(iters)):
        fig = plt.figure(figsize=(6, 7))
        ax = plt.gca()

        radius = (iradius, oradius)
        if args.dark:
            color = 'white'
        else:
            color = 'black'
        inner = plt.Circle(
            center, radius[0], color=color, fill=False)
        outer = plt.Circle(
            center, radius[1], color=color, fill=False)
        ax.add_artist(outer)

        z = np.zeros([bins, bins])

        for fidx in range(p.shape[1] // 2):
            X = []
            X.append(pos_t_1[t, fidx * 2])
            X.append(pos_t_1[t, fidx * 2 + 1])
            X.append(vel_t_1[t, fidx * 2])
            X.append(vel_t_1[t, fidx * 2 + 1])

            Y = []
            Y.append(vel_t[t, fidx * 2])
            Y.append(vel_t[t, fidx * 2 + 1])
            
            for nidx in range(p.shape[1] // 2):
                if nidx == fidx:
                    continue    
                X.append(pos_t_1[t, nidx * 2])
                X.append(pos_t_1[t, nidx * 2 + 1])
                X.append(vel_t_1[t, nidx * 2])
                X.append(vel_t_1[t, nidx * 2 + 1])

            X = np.array([X])
            Y = np.array([Y])

            prediction = np.array(model.predict(X))

            def logbound(val, max_logvar=0, min_logvar=-10):
                logsigma = max_logvar - \
                    np.log(np.exp(max_logvar - val) + 1)
                logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
                return logsigma

            prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
            prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

            for _ in range(args.sample_size):
                sample_velx = np.random.normal(prediction[0, 0], prediction[0, 2], 1)[0]
                sample_vely = np.random.normal(prediction[0, 1], prediction[0, 3], 1)[0]

                x_hat = pos_t_1[t, fidx * 2] + sample_velx * args.timestep
                y_hat = pos_t_1[t, fidx * 2 + 1] + sample_vely * args.timestep

                dist_x = np.abs(np.array(x_hat - m_x[:, 0]))
                dist_y = np.abs(np.array(y_hat - m_y[0, :]))
                min_xidx = np.argmin(dist_x)
                min_yidx = np.argmin(dist_y)
                z[min_xidx, min_yidx] += 1

            if not args.fish_like:
                ax.add_artist(plt.Circle((pos_t_1[t, fidx * 2], pos_t_1[t, fidx * 2 + 1]), 0.01, color='white', fill=False))
            ax.add_artist(plt.Circle((pos_t[t, fidx * 2], pos_t[t, fidx * 2 + 1]), 0.01, color='green', fill=False))

            if args.fish_like:
                phi = np.arctan2(vel_t_1[t, fidx * 2 + 1], vel_t_1[t, fidx * 2]) * 180 / np.pi
                rotated_img = image.rotate(phi)
                ax.imshow(rotated_img, extent=[pos_t_1[t, fidx * 2] - 0.03, pos_t_1[t, fidx * 2] + 0.03, pos_t_1[t, fidx * 2 + 1] -
                                                0.03, pos_t_1[t, fidx * 2 + 1] + 0.03], aspect='equal', zorder=1)
       
        z /= (iters * (p.shape[1] // 2))
        z_min, z_max = 0, 0.0011

        palette = sns.color_palette('RdYlBu_r', 1000)
        palette = [(0, 0, 0, 0)] + palette
        sns.set_palette(palette)
        palette = sns.color_palette()
        cmap = ListedColormap(palette.as_hex())

        c = ax.pcolormesh(m_x, m_y, z, cmap=cmap, vmin=z_min, vmax=z_max)
        fig.colorbar(c, ax=ax, label='Cell occupancy (%)', orientation='horizontal', pad=0.05)

        # ax.axis('off')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        plt.tight_layout()
        png_fname = args.out_dir + '/' + str(t).zfill(6)
        plt.savefig(
            str(png_fname) + '.png',
            transparent=True,
            dpi=300
        )
        plt.close('all')


def polar_sim(model, setup, args):
    global radius, center

    inputs = None
    outputs = None
    p = np.loadtxt(args.reference)
    vel = np.loadtxt(args.reference.replace('positions', 'velocities'))
    timestep = args.timestep

    pos_t = np.roll(p, shift=1, axis=0)[2:, :]
    rad_t = np.sqrt( (pos_t[:, 0] - setup.center()[0]) ** 2 + (pos_t[:, 1] - setup.center()[1]) ** 2)

    pos_t_1 = np.roll(p, shift=1, axis=0)[1:-1, :]
    rad_t_1 = np.sqrt( (pos_t_1[:, 0] - setup.center()[0]) ** 2 + (pos_t_1[:, 1] - setup.center()[1]) ** 2)

    vel_t = (p - np.roll(p, shift=1, axis=0))[2:, :] / timestep
    radial_vel_t = (pos_t[:, 0] * vel_t[:, 1] - pos_t[:, 1] * vel_t[:, 0]) / (pos_t[:, 0] ** 2 + pos_t[:, 1] ** 2)
    hdg_t = np.array(list(map(angle_to_pipi, np.arctan2(vel_t[:, 1], vel_t[:, 0]))))

    vel_t_1 = (p - np.roll(p, shift=1, axis=0))[1:-1, :] / timestep
    radial_vel_t_1 = (pos_t_1[:, 0] * vel_t_1[:, 1] - pos_t_1[:, 1] * vel_t_1[:, 0]) / (pos_t_1[:, 0] ** 2 + pos_t_1[:, 1] ** 2)
    hdg_t_1 = np.array(list(map(angle_to_pipi, np.arctan2(vel_t_1[:, 1], vel_t_1[:, 0]))))

    X = np.array([rad_t_1, np.cos(hdg_t_1), np.sin(hdg_t_1), vel_t_1[:, 0], vel_t_1[:, 1]])
    Y = np.array([(rad_t-rad_t_1) / timestep, radial_vel_t])
    inputs = X
    outputs = Y

    X = X.transpose()
    Y = Y.transpose()

    if args.iterations < 0:
        iters = p.shape[0]
    else:
        iters = args.iterations


    bins = 1000
    m_y, m_x = np.meshgrid(np.linspace(center[0] - (oradius + 0.0001),
                                   center[0] + (oradius + 0.0001), bins),
                       np.linspace(center[1] - (oradius + 0.0001),
                                   center[1] + (oradius + 0.0001), bins))
    r = np.sqrt((m_x - center[0]) ** 2 + (m_y - center[1]) ** 2)
    outside_els = np.sum(r > radius[1])


    for t in tqdm.tqdm(range(iters)):
        fig = plt.figure(figsize=(6, 7))
        ax = plt.gca()

        radius = (iradius, oradius)
        if args.dark:
            color = 'white'
        else:
            color = 'black'
        inner = plt.Circle(
            center, radius[0], color=color, fill=False)
        outer = plt.Circle(
            center, radius[1], color=color, fill=False)
        ax.add_artist(outer)

        z = np.zeros([bins, bins])

        prediction = np.array(model.predict(X[t].reshape(1, X.shape[1])))

        def logbound(val, max_logvar=0, min_logvar=-10):
            logsigma = max_logvar - \
                np.log(np.exp(max_logvar - val) + 1)
            logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
            return logsigma

        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))


        for _ in range(args.sample_size):
            sample_velx = np.random.normal(prediction[0, 0], prediction[0, 2], 1)[0]
            sample_vely = np.random.normal(prediction[0, 1], prediction[0, 3], 1)[0]

            x_hat = pos_t_1[t, 0] + sample_velx * args.timestep
            y_hat = pos_t_1[t, 1] + sample_vely * args.timestep

            dist_x = np.abs(np.array(x_hat - m_x[:, 0]))
            dist_y = np.abs(np.array(y_hat - m_y[0, :]))
            min_xidx = np.argmin(dist_x)
            min_yidx = np.argmin(dist_y)
            z[min_xidx, min_yidx] += 1

        z /= iters
        z_min, z_max = 0, 0.0011

        if not args.fish_like:
            ax.add_artist(plt.Circle((pos_t_1[t, 0], pos_t_1[t, 1]), 0.01, color='white', fill=False))
        ax.add_artist(plt.Circle((pos_t[t, 0], pos_t[t, 1]), 0.01, color='green', fill=False))

        if args.fish_like:
            phi = np.arctan2(vel[t, 1], vel[t, 0]) * 180 / np.pi
            rotated_img = image.rotate(phi)
            ax.imshow(rotated_img, extent=[pos_t_1[t, 0] - 0.03, pos_t_1[t, 0] + 0.03, pos_t_1[t, 1] -
                                            0.03, pos_t_1[t, 1] + 0.03], aspect='equal', zorder=1)

        palette = sns.color_palette('RdYlBu_r', 1000)
        palette = [(0, 0, 0, 0)] + palette
        sns.set_palette(palette)
        palette = sns.color_palette()
        cmap = ListedColormap(palette.as_hex())

        c = ax.pcolormesh(m_x, m_y, z, cmap=cmap, vmin=z_min, vmax=z_max)
        fig.colorbar(c, ax=ax, label='Cell occupancy (%)', orientation='horizontal', pad=0.05)

        # ax.axis('off')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        plt.tight_layout()
        png_fname = args.out_dir + '/' + str(t).zfill(6)
        plt.savefig(
            str(png_fname) + '.png',
            transparent=True,
            dpi=300
        )
        plt.close('all')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the decision heatmap for probabilistic models')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--reference', '-r', type=str,
                        help='Path to a reference experiment position file',
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--polar', action='store_true',
                        help='Use polar inputs instead of cartesian coordinates',
                        default=False)
    parser.add_argument('--model', '-m', type=str,
                        help='Model file name to use',
                        required=True)
    parser.add_argument('--iterations', '-i', type=int,
                        help='Number of iteration of the simulation',
                        required=False,
                        default=-1)
    parser.add_argument('--sample-size', '-s', type=int,
                        help='Samples to draw for the velocity distribution',
                        required=False,
                        default=1000)
    parser.add_argument('--dark', action='store_true',
                        help='Render dark friendly icons',
                        default=False)
    parser.add_argument('--out-dir', '-o', type=str,
                        help='Output directory name',
                        required=True)
    parser.add_argument('--fish-like', action='store_true',
                        help='Images instead of points',
                        default=False)
    args = parser.parse_args()


    model = tf.keras.models.load_model(Path(args.path).joinpath(args.model + '_model.h5'), custom_objects={
        'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})
    setup = CircularCorridor()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if not args.polar:
        cart_sim(model, setup, args)
    else:
        polar_sim(model, setup, args)
