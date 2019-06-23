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
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('dark_background')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions')
    parser.add_argument('--positions', '-p', type=str,
                        help='Path to the trajectory file',
                        required=True)
    parser.add_argument('--fname', '-o', type=str,
                        help='output file name',
                        required=True)
    parser.add_argument('--ind', '-i', type=int,
                        default=-1,
                        help='single individual id to plot',
                        required=True)
    parser.add_argument('--timesteps', '-t', type=int,
                        default=-1,
                        help='Timesteps to use in the plot',
                        required=False)
    parser.add_argument('--points',  action='store_true',
                        help='Plot points instead of trajectory lines', default=False)
    parser.add_argument('--open',  action='store_true',
                        help='No probabilities are contained within the trajectory file', default=False)
    args = parser.parse_args()

    traj = np.loadtxt(args.positions)
    tsteps = traj.shape[0]
    if args.timesteps < 0:
        args.timesteps = tsteps
    individuals = int(traj.shape[1] / 2)

    iradius = 0.655172413793
    oradius = 1.0
    center = (0, 0)
    radius = (iradius, oradius)

    fig = plt.figure(figsize=(7, 7))
    ax = plt.gca()

    inner = plt.Circle(
        center, radius[0], color='white', fill=False)
    outer = plt.Circle(
        center, radius[1], color='white', fill=False)
    if not args.open:
        ax.add_artist(inner)
    ax.add_artist(outer)

    if args.ind < 0:
        for j in range(int(traj.shape[1] / 2)):
            if not args.points:
                plt.plot(traj[:args.timesteps, j*2],
                         traj[:args.timesteps, j*2 + 1], linewidth=0.2)
            else:
                warnings.warn(
                    'Not supported for all individuals. Please specify an index')
    else:
        if not args.points:
            plt.plot(traj[:args.timesteps, args.ind*2],
                     traj[:args.timesteps, args.ind*2 + 1], linewidth=0.2)
        else:
            c = plt.scatter(traj[:args.timesteps, args.ind*2],
                        traj[:args.timesteps, args.ind*2 + 1], 1, vmin=0, vmax=0.5, cmap='YlOrRd')
            # fig.colorbar(c, ax=ax, label='Instantaneous velocity (m/s)', orientation='horizontal', pad=-0.07)
            
    # ax.axis('off')
    # ax.set_xlim((0.25, 0.9))
    # ax.set_ylim((0.2, 0.9))
    plt.tight_layout()
    plt.savefig(args.fname + '.png', dpi=300)
