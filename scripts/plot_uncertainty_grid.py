#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
import socket
import warnings
import argparse
import datetime
import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.misc
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, TransformedBbox
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import seaborn as sns
from pprint import pprint
from matplotlib.colors import ListedColormap

plt.style.use('dark_background')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot the uncertainty grid for an experiment')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the trajectory file',
                        required=True)
    parser.add_argument('--fname', '-o', type=str,
                        help='output file name',
                        required=True)
    parser.add_argument('--timesteps', '-t', type=int,
                        default=-1,
                        help='Timesteps to use in the plot',
                        required=False)
    parser.add_argument('--open',  action='store_true',
                        help='No probabilities are contained within the trajectory file', default=False)

    args = parser.parse_args()

    traj = np.loadtxt(args.path)
    sigmas = np.loadtxt(args.path.replace('positions', 'sigmas'))

    tsteps = traj.shape[0]

    if args.timesteps < 0:
        args.timesteps = tsteps
    individuals = int(traj.shape[1] / 2)

    traj = traj[:args.timesteps, :]
    sigmas = sigmas[:args.timesteps, :]

    iradius = 0.655172413793
    oradius = 1.0
    center = (0, 0)
    radius = (iradius, oradius)

    bins = 120
    y, x = np.meshgrid(np.linspace(center[0]-(oradius + 0.0001),
                                   center[0]+(oradius + 0.0001), bins),
                       np.linspace(center[1]-(oradius + 0.0001),
                                   center[1]+(oradius + 0.0001), bins))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    outside_els = np.sum(r > radius[1])

    z = np.ones([bins, bins])
    ux = np.zeros([bins, bins])
    uy = np.zeros([bins, bins])

    idcs = range(individuals)

    for i in range(args.timesteps):
        for j in idcs:
            traj_x = traj[i, j*2]
            traj_y = traj[i, j*2+1]

            dist_x = np.abs(np.array(traj_x - x[:, 0]))
            dist_y = np.abs(np.array(traj_y - y[0, :]))
            min_xidx = np.argmin(dist_x)
            min_yidx = np.argmin(dist_y)
            z[min_xidx, min_yidx] += 1
            ux[min_xidx, min_yidx] += sigmas[i, 0]
            uy[min_xidx, min_yidx] += sigmas[i, 1]

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ux[i, j] /= z[i, j]
            uy[i, j] /= z[i, j]

    z_min, z_max = 0, np.max([np.max(ux), np.max(uy)])

    def plot_grid(z, z_min, z_max, args, sname):
        fig = plt.figure(figsize=(6, 7))
        ax = plt.gca()

        inner = plt.Circle(
            center, radius[0], color='white', fill=False)
        outer = plt.Circle(
            center, radius[1], color='white', fill=False)
        if not args.open:
            ax.add_artist(inner)
        ax.add_artist(outer)

        palette = sns.color_palette('RdYlBu_r', 1000)
        palette = [(0, 0, 0, 0)] + palette 
        sns.set_palette(palette)
        palette = sns.color_palette()
        cmap = ListedColormap(palette.as_hex())

        c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
        fig.colorbar(c, ax=ax, label='Average sigma value', orientation='horizontal', pad=0.05)

        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        plt.tight_layout()
        plt.savefig(args.fname + '_' + sname + '.png', dpi=300)

    plot_grid(ux, z_min, z_max, args, 'x')
    plot_grid(uy, z_min, z_max, args, 'y')

