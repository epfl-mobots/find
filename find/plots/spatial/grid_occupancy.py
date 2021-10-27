#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

from find.plots.common import *

import matplotlib
from scipy.interpolate import griddata
import scipy.stats as st


def occupancy_grid(data, fig, type, ax, args, pad=0.05):
    outer = plt.Circle((0, 0), args.radius * 1.005,
                       color='black', fill=False)
    ax.add_artist(outer)

    y, x = np.meshgrid(np.linspace(args.center[0] - (args.radius + 0.0001),
                                   args.center[0] + (args.radius + 0.0001), args.grid_bins),
                       np.linspace(args.center[1] - (args.radius + 0.0001),
                                   args.center[1] + (args.radius + 0.0001), args.grid_bins))
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
    z /= total_steps
    z *= 100

    # we need a custom paletter for this plot
    palette = sns.color_palette('RdYlBu_r', 1000)
    palette = [(1, 1, 1, 0.5)] + palette
    sns.set_palette(palette)
    palette = sns.color_palette()
    cmap = ListedColormap(palette.as_hex())

    lb, ub = 0.0, 0.02
    step = 0.005

    c = ax.pcolormesh(x, y, z, cmap=cmap,
                      shading='gouraud', vmin=lb, vmax=ub, alpha=1.0)
    cbar = fig.colorbar(c, ax=ax, label='Cell occupancy (%)',
                        orientation='horizontal', pad=pad, extend='max')

    cbar.set_ticks(np.arange(lb, ub + 0.001, step))
    cbar.set_ticklabels(np.arange(lb, ub * 100 + 0.001, step * 100))

    ax.set_yticks(np.arange(-args.radius,
                            args.radius + 0.001, args.radius / 2))
    ax.set_xticks(np.arange(-args.radius,
                            args.radius + 0.001, args.radius / 2))
    ax.set_xlim([-(args.radius * 1.05), args.radius * 1.05])
    ax.set_ylim([-(args.radius * 1.05), args.radius * 1.05])
    ax.set_title(type)
    return ax


def occupancy_grid_dist(data, fig, type, ax, args, pad=0.05):
    grid = {}
    xy = np.empty((0, 2))
    for traj in data[type]:
        tsteps = traj.shape[0]
        individuals = traj.shape[1] // 2
        idcs = range(individuals)
        for i in range(tsteps):
            for j in idcs:
                traj_x = traj[i, j * 2]
                traj_y = traj[i, j * 2 + 1]
                xy = np.vstack((xy, np.array([traj_x, traj_y]).T))

    cmap = matplotlib.cm.get_cmap('jet')
    _ = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    xx, yy = np.mgrid[-0.3:0.3:300j, -0.3:0.3:300j]
    kernel = st.gaussian_kde(xy.T)
    grid_pos = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(grid_pos).T, xx.shape)
    ax.contourf(xx, yy, f, cmap=cmap)
    outer = plt.Circle(
        (0, 0), 0.25, color='k', fill=False)
    ax.add_artist(outer)
    ax.set_xlim([-0.3, 0.3])
    ax.set_ylim([-0.3, 0.3])
    grid['xx'] = xx
    grid['yy'] = yy
    grid['f'] = f
    return ax, grid


def plot_grid_differences(grids, path, args):
    if 'Real' not in grids.keys() and ('Hybrid' not in grids.keys() or 'Virtual' not in grids.keys()):
        import warnings
        warnings.warn('Skipping grid difference plots')
        return

    r_xx = grids['Real']['xx']
    r_yy = grids['Real']['yy']
    r_f = grids['Real']['f']

    keys = list(grids.keys())
    keys.remove('Real')
    for k in keys:
        xx_diff = r_xx - grids[k]['xx']
        yy_diff = r_yy - grids[k]['yy']
        f_diff = r_f - grids[k]['f']

        cmap = matplotlib.cm.get_cmap('jet')
        _ = plt.figure(figsize=(6, 6))
        ax = plt.gca()

        ax.contourf(r_xx, r_yy, f_diff, cmap=cmap)
        outer = plt.Circle(
            (0, 0), 0.25, color='k', fill=False)
        ax.add_artist(outer)
        ax.set_xlim([-0.3, 0.3])
        ax.set_ylim([-0.3, 0.3])
        plt.savefig(path + '/test_grid.png')
        plt.close()


def plot(exp_files, path, args):
    grids = {}
    for k, v in exp_files.items():
        data = {}
        data[k] = []
        files = glob.glob(args.path + '/' + v)
        for f in files:
            data[k].append(np.loadtxt(f) * args.radius)

        fig = plt.figure(figsize=(6, 7))
        ax = plt.gca()
        ax = occupancy_grid(data, fig, k, ax, args)
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        plt.savefig(path + '/occupancy_' + k.lower())
        plt.close()

        fig = plt.figure(figsize=(6, 7))
        ax = plt.gca()
        ax, g = occupancy_grid_dist(data, fig, k, ax, args)
        grids[k] = g
        plt.grid(linestyle='dotted')
        plt.tight_layout()
        plt.savefig(path + '/occupancy_dist_' + k.lower())
        plt.close()

    plot_grid_differences(grids, path, args)


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
                        default=300,
                        required=False)
    parser.add_argument('--center',
                        type=float,
                        nargs='+',
                        help='The centroidal coordinates for the setups used',
                        default=[0.0, 0.0],
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
