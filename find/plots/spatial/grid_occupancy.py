#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

from find.plots.common import *

import matplotlib
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter


def construct_grid(data, type, args):
    y, x = np.meshgrid(np.linspace(args.center[0] - (args.radius + 0.01),
                                   args.center[0] + (args.radius + 0.01), args.grid_bins),
                       np.linspace(args.center[1] - (args.radius + 0.01),
                                   args.center[1] + (args.radius + 0.01), args.grid_bins))
    z = np.zeros([args.grid_bins, args.grid_bins])

    total_steps = 0
    for traj in data[type]:
        tsteps = traj.shape[0]
        total_steps += tsteps
        individuals = traj.shape[1] // 2
        idcs = range(individuals)

        for i in range(tsteps):
            for j in idcs:
                traj_x = traj[i, j * 2]
                traj_y = traj[i, j * 2 + 1]
                dist_x = np.abs(np.array(traj_x - x[:, 0]))
                dist_y = np.abs(np.array(traj_y - y[0, :]))
                min_xidx = np.argmin(dist_x)
                min_yidx = np.argmin(dist_y)
                z[min_xidx, min_yidx] += 1
    z /= 2 * total_steps
    z *= 100

    if type == 'Real' or type == 'Hybrid':
        # here there is a stationary instance which seems to be a data issue
        # we smooth approximately 0.07% instances that are abnormally big (perhaps tracking error)
        z[z > 0.0045] = np.mean(z)
    print('Occupancy grid computed for type: {}'.format(type))

    if args.grid_smooth:
        z = gaussian_filter(z, sigma=5.0)

    return x, y, z


def occupancy_grid(data, grid, fig, type, ax, args, draw_colorbar=True, pad=0.1):
    x, y, z = grid['x'], grid['y'], grid['z']

    cutoff_val = args.grid_cutoff_val
    if cutoff_val < 0:
        step = 0.0005
        cutoff = list(np.arange(step, np.max(z) + step / 2, step))
        cutoff_val = step
        for i in range(len(cutoff)):
            c = cutoff[i]
            if np.sum(np.array(z > c)) / np.size(z) > args.grid_cutoff_thres:
                if i+1 < len(cutoff):
                    cutoff_val = cutoff[i+1]
                else:
                    cutoff_val = cutoff[i]
    lb, ub = np.min(z), cutoff_val

    # we need a custom palette for this plot
    # palette = sns.color_palette('viridis', args.grid_bins * args.grid_bins)
    # cmap = ListedColormap(palette.as_hex())
    cmap = matplotlib.cm.get_cmap('jet')

    c = ax.pcolormesh(x, y, z, cmap=cmap, shading='auto',
                      vmin=lb, vmax=ub, alpha=1.0)

    if draw_colorbar:
        fig.colorbar(c, ax=ax, label='Cell occupancy (%)',
                     location='left', pad=pad, extend='max')

    ax.set_yticks(np.arange(-args.radius,
                            args.radius + 0.001, args.radius / 2))
    ax.set_xticks(np.arange(-args.radius,
                            args.radius + 0.001, args.radius / 2))
    ax.set_xlim([-(args.radius * 1.05), args.radius * 1.05])
    ax.set_ylim([-(args.radius * 1.05), args.radius * 1.05])
    ax.set_title(type)

    outer = plt.Circle((0, 0), args.radius * 1.0005,
                       color='white', fill=False)
    ax.add_artist(outer)

    return ax


def plot_grid_differences(grids, path, args, draw_colorbar=True, pad=0.1):
    if 'Real' not in grids.keys() and ('Hybrid' not in grids.keys() or 'Virtual' not in grids.keys()):
        import warnings
        warnings.warn('Skipping grid difference plots')
        return

    r_x = grids['Real']['x']
    r_y = grids['Real']['y']
    r_z = grids['Real']['z']

    keys = list(grids.keys())
    keys.remove('Real')
    for k in keys:
        z_diff = r_z - grids[k]['z']

        cmap = matplotlib.cm.get_cmap('jet')

        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()

        print(np.max(np.abs(z_diff)))

        vmax = args.grid_cutoff_val
        if vmax < 0:
            vmax = np.max(z_diff)

        c = ax.pcolormesh(r_x, r_y, np.abs(z_diff), cmap=cmap, shading='auto',
                          vmin=0,
                          vmax=vmax, alpha=1.0
                          )

        if draw_colorbar:
            fig.colorbar(c, ax=ax, label='Cell occupancy (%)',
                         location='left', pad=pad, extend='max')

        ax.set_yticks(np.arange(-args.radius,
                                args.radius + 0.001, args.radius / 2))
        ax.set_xticks(np.arange(-args.radius,
                                args.radius + 0.001, args.radius / 2))
        ax.set_xlim([-(args.radius * 1.05), args.radius * 1.05])
        ax.set_ylim([-(args.radius * 1.05), args.radius * 1.05])
        ax.set_title(k)

        outer = plt.Circle((0, 0), args.radius * 1.0005,
                           color='white', fill=False)
        ax.add_artist(outer)
        plt.tight_layout()
        plt.savefig(path + '/occupancy_dist_{}.png'.format(k))
        plt.close()


def plot(exp_files, path, args):
    grids = {}
    for k, v in exp_files.items():
        data = {}
        data[k] = []
        files = glob.glob(args.path + '/' + v)
        for f in files:
            data[k].append(np.loadtxt(f) * args.radius)
        print('Done loading data for type: {}'.format(k))

        x, y, z = construct_grid(data, k, args)
        grid = {'x': x, 'y': y, 'z': z}
        grids[k] = grid

        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax = occupancy_grid(data, grid, fig, k, ax, args, pad=0.13)
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        plt.savefig(path + '/occupancy_{}.png'.format(k))
        plt.close()

    plot_grid_differences(grids, path, args, pad=0.135)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions')
    parser.add_argument('--positions', '-p', type=str,
                        help='Path to the trajectory file',
                        required=True)
    parser.add_argument('--type',
                        nargs='+',
                        default=['Real', 'Hybrid', 'Virtual'],
                        choices=['Real', 'Hybrid', 'Virtual'])
    parser.add_argument('--original_files',
                        type=str,
                        default='raw/*processed_positions.dat',
                        required=False)
    parser.add_argument('--hybrid_files',
                        type=str,
                        default='generated/*generated_positions.dat',
                        required=False)
    parser.add_argument('--virtual_files',
                        type=str,
                        default='generated/*generated_virtu_positions.dat',
                        required=False)
    parser.add_argument('--radius',
                        type=float,
                        help='Radius',
                        default=0.25,
                        required=False)
    parser.add_argument('--grid_bins',
                        type=int,
                        help='Number of bins for the occupancy grid plot',
                        default=208,
                        required=False)
    parser.add_argument('--center',
                        type=float,
                        nargs='+',
                        help='The centroidal coordinates for the setups used',
                        default=[0.0, 0.0],
                        required=False)
    parser.add_argument('--grid_smooth',
                        action='store_true',
                        help='Smooth the grid for visual reasons if true',
                        default=False,
                        required=False)
    parser.add_argument('--grid_cutoff_thres',
                        type=float,
                        help='Cutoff point threshold for the percentage of points that are allowed to be removed to not squash the grid drawing colours',
                        default=0.05,
                        required=False)
    parser.add_argument('--grid_cutoff_val',
                        type=float,
                        help='Force the cutoff value of the grid for consistency (overrides grid_cutoff_thres)',
                        default=-1,
                        required=False)
    args = parser.parse_args()

    exp_files = {}
    for t in args.type:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
