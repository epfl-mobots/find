#!/usr/bin/env python
import os
import re
import glob
import random
import pickle
import argparse
import numpy as np
from scipy.stats import t

from rich.progress import track, Progress
from rich.console import Console

from find.utils.features import Velocities

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, InsetPosition
from scipy.ndimage.filters import gaussian_filter


def plot_phase_diagram(data, path, args):
    for e in data.keys():
        ops = []
        ors = []
        for i in range(len(data[e]['pos'])):
            pos = data[e]['pos'][i]
            vel = data[e]['vel'][i]
            num_inds = pos.shape[1] // 2

            coms = np.zeros(shape=(pos.shape[0], 2))
            for n in range(num_inds):
                coms[:, 0] += pos[:, n*2]
                coms[:, 1] += pos[:, n*2+1]
            coms /= num_inds

            uis = np.zeros(shape=(pos.shape[0], 2))
            ris = np.zeros(shape=(pos.shape[0], 2))
            for n in range(num_inds):
                norms_uis = np.sqrt(vel[:, n*2+1] ** 2 + vel[:, n*2] ** 2) + 0.0001
                ui = np.array(
                    [vel[:, n*2] / norms_uis, vel[:, n*2+1] / norms_uis]).T
                uis += ui

                norms_ris = np.sqrt((coms[:, 1] - pos[:, n*2+1]) ** 2
                                + (coms[:, 0] - pos[:, n*2]) ** 2) + 0.0001
                ri = np.array(
                    [(coms[:, 0] - pos[:, n*2]) * vel[:, n*2], 
                    (coms[:, 1] - pos[:, n*2+1]) * vel[:, n*2+1]]).T
                ri[:, 0] /= (abs(norms_ris) * norms_uis)
                ri[:, 1] /= (abs(norms_ris) * norms_uis)
                ris += ri
            uis /= num_inds
            ris /= num_inds
            ops += (np.sqrt(uis[:, 0] ** 2 + uis[:, 1]
                            ** 2)).tolist()
            ors += (np.sqrt(ris[:, 0] ** 2 + ris[:, 1]
                            ** 2)).tolist()

        num_bins = 100
        y, x = np.meshgrid(np.linspace(0, 1, num_bins),
                           np.linspace(0, 1, num_bins))
        z = np.zeros([num_bins, num_bins])

        for it in range(len(ops)):
            dist_x = np.abs(np.array(ors[it] - x[:, 0]))
            dist_y = np.abs(np.array(ops[it] - y[0, :]))
            # dist_x = np.abs(np.array(ops[it] - x[:, 0]))
            # dist_y = np.abs(np.array(ors[it] - y[0, :]))
            min_xidx = np.argmin(dist_x)
            min_yidx = np.argmin(dist_y)
            z[min_xidx, min_yidx] += 1
        z /= len(ops)
        z *= 100

        if args.grid_smooth:
            z = gaussian_filter(z, sigma=3.0)

        lb, ub = 0, 0.04
        # lb, ub = 0, np.max(z)
        print('z max: {}', np.max(z))
        print('z mean: {}', np.mean(z))

        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()

        cmap = matplotlib.cm.get_cmap('jet')
        colors = cmap(np.linspace(0, 1, num_bins**2))
        new_colors = np.vstack(([matplotlib.colors.to_rgba('black')], colors))
        cmap = matplotlib.colors.ListedColormap(new_colors)

        c = ax.pcolormesh(x, y, z, cmap=cmap, shading='auto',
                          vmin=lb, vmax=ub, alpha=1.0)

        # if draw_colorbar:
        fig.colorbar(c, ax=ax, label='Cell occupancy (%)',
                     location='left', pad=0.1, extend='max')

        # plt.plot(ors, ops, linestyle='None', marker='.')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal', 'box')

        # plt.show()
        plt.savefig(path+'{}-phase_diagram.png'.format(e), dpi=300)


def plot(exp_files, path, args):
    data = {}
    grids = {}
    num_inds = -1

    for e in sorted(exp_files.keys()):
        samples = 0

        if e == 'BOBI' or 'Simu' in e:
            timestep = args.bt
        elif e == 'F44':
            timestep = args.f44t
        else:
            timestep = args.timestep

        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        if args.robot:
            data[e]['ridx'] = []

        for p in pos:
            if e == 'Virtual (Toulouse)':
                f = open(p)
                # to allow for loading fortran's doubles
                strarray = f.read().replace("D+", "E+").replace("D-", "E-")
                f.close()
                num_ind = len(strarray.split('\n')[0].strip().split('  '))
                positions = np.fromstring(
                    strarray, sep='\n').reshape(-1, num_ind) * args.radius
            elif e == 'Virtual (Toulouse cpp)':
                positions = np.loadtxt(p)[:, 2:] * args.radius
            else:
                positions = np.loadtxt(p) * args.radius
            velocities = Velocities([positions], timestep).get()[0]

            samples += positions.shape[0]
            num_inds = positions.shape[1] // 2

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

        plot_phase_diagram(data, path, args)
