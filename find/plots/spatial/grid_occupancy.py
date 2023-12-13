#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

from find.plots.common import *

import matplotlib
import scipy.stats as st
from scipy.ndimage.filters import gaussian_filter


def construct_grid_sep(data, type, args, sigma=5.0, ridcs=None):
    num_inds = data[type][0].shape[1] // 2
    x = [None] * num_inds
    y = [None] * num_inds
    z = [None] * num_inds

    order = [list(range(num_inds)) for l in range(len(data[type]))]
    if args.robot and ridcs is not None:
        for fn in range(len(order)):
            ridx = ridcs[type][fn]
            if ridx >= 0:
                order[fn].remove(ridx)
                order[fn] = [ridx] + order[fn]

    ndata = {}
    for i in range(num_inds):
        ndata[i] = {}
        ndata[i][type] = []

    for fn, o in enumerate(order):
        traj = data[type][fn]
        for idx in o:
            ndata[idx][type].append(traj[:, (idx * 2):(idx * 2 + 2)])

    for ind in range(num_inds):
        x[ind], y[ind], z[ind] = construct_grid(ndata[ind], type, args, sigma)

    return x, y, z


def construct_grid(data, type, args, sigma=5.0):
    y, x = np.meshgrid(np.linspace(args.center[0] - (args.radius + 0.5),
                                   args.center[0] + (args.radius + 0.5), args.grid_bins),
                       np.linspace(args.center[1] - (args.radius + 0.5),
                                   args.center[1] + (args.radius + 0.5), args.grid_bins))
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
    z /= (data[type][0].shape[1] // 2) * total_steps
    z *= 100

    if type == 'Real' or type == 'Hybrid':
        # here there is a stationary instance which seems to be a data issue
        # we smooth approximately 0.07% instances that are abnormally big (perhaps tracking error)
        z[z > 0.0045] = np.mean(z)
    print('Occupancy grid computed for type: {}'.format(type))

    if args.grid_smooth:
        z = gaussian_filter(z, sigma=sigma)

    return x, y, z


def occupancy_grid(data, grid, fig, type, ax, args, draw_colorbar=True, draw_circle=True, pad=0.1):
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
    print('z max: {}', np.max(z))
    print('z mean: {}', np.mean(z))

    # we need a custom palette for this plot
    # palette = sns.color_palette('viridis', args.grid_bins * args.grid_bins)
    # cmap = ListedColormap(palette.as_hex())
    # cmap = matplotlib.cm.get_cmap('jet', 15)
    cmap = matplotlib.cm.get_cmap('jet')

    c = ax.pcolormesh(x, y, z, cmap=cmap, shading='auto',
                      #   vmin=lb, vmax=ub,
                      alpha=1.0)

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

    if draw_circle:
        outer = plt.Circle((0, 0), args.radius * 1.0005,
                           color='white', fill=False)
        ax.add_artist(outer)
    ax.set_aspect('equal', 'box')

    return ax, c


def grid_difference(grids, type1, type2, fig, ax, args, title=None, draw_colorbar=True, draw_circle=True, pad=0.1):
    cmap = matplotlib.cm.get_cmap('jet')

    r_x = grids[type1]['x']
    r_y = grids[type1]['y']
    r_z = grids[type1]['z']
    z_diff = r_z - grids[type2]['z']

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

    if type1 == 'Virtual (Toulouse)':
        type1 = 'ABC'
    if type2 == 'Virtual (Toulouse)':
        type2 = 'ABC'

    if type1 == 'Virtual':
        type1 = 'HR-NNig'
    if type2 == 'Virtual':
        type2 = 'HR-NNig'

    if type1 == 'Real':
        type1 = 'CD'
    if type2 == 'Real':
        type2 = 'CD'

    if title is None:
        ax.set_title('|{} - {}|'.format(type1, type2))
    else:
        ax.set_title('{}'.format(title))

    if draw_circle:
        outer = plt.Circle((0, 0), args.radius * 1.0005,
                           color='white', fill=False)
        ax.add_artist(outer)
    ax.set_aspect('equal', 'box')

    return ax, c


def plot(exp_files, path, args):
    grids = {}
    for k, v in exp_files.items():
        data = {}
        data[k] = []
        files = glob.glob(args.path + '/' + v)
        for f in files:
            data[k].append(np.loadtxt(f) * args.radius)
        print('Done loading data for type: {}'.format(k))

        # print(len(data[k]))
        # fig = plt.figure(figsize=(6, 5))
        # for d in data[k]:
        #     plt.plot(d[:, 0], d[:, 1], linestyle='None',
        #              marker='.', markersize=1)
        # outer = plt.Circle((0, 0), 0.25,
        #                    color='black', fill=False)
        # ax = plt.gca()
        # ax.set_xlim([-0.25, 0.25])
        # ax.set_ylim([-0.25, 0.25])
        # ax.add_artist(outer)
        # ax.set_aspect('equal', 'box')
        # plt.show()
        # exit(1)

        if not args.separate:
            x, y, z = construct_grid(data, k, args)
            grid = {'x': x, 'y': y, 'z': z}
            grids[k] = grid

            fig = plt.figure(figsize=(6, 5))
            ax = plt.gca()
            ax, _ = occupancy_grid(data, grid, fig, k, ax, args, pad=0.13)
            plt.grid(linestyle='dotted')
            plt.tight_layout()
            plt.savefig(path + '/occupancy_{}.png'.format(k),
                        bbox_inches='tight')
            plt.close()
        else:
            x, y, z = construct_grid_sep(data, k, args)
            for i in range(len(x)):
                grid = {'x': x[i], 'y': y[i], 'z': z[i]}
                grids[k] = grid

                fig = plt.figure(figsize=(6, 5))
                ax = plt.gca()
                ax, _ = occupancy_grid(data, grid, fig, k, ax, args, pad=0.13)
                plt.grid(linestyle='dotted')
                plt.tight_layout()
                plt.savefig(path + '/occupancy_{}-idx_{}.png'.format(k, i),
                            bbox_inches='tight')
                plt.close()

    if 'Real' in grids.keys() and ('Hybrid' in grids.keys() or 'Virtual' in grids.keys()):
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax, _ = grid_difference(
            grids, 'Real', 'Virtual', fig, ax, args, pad=0.135)
        plt.tight_layout()
        plt.savefig(
            path + '/occupancy_diff_{}-{}.png'.format('Real', 'Virtual'), bbox_inches='tight')
        plt.close()

        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax, _ = grid_difference(
            grids, 'Real', 'Hybrid', fig, ax, args, pad=0.135)
        plt.tight_layout()
        plt.savefig(path + '/occupancy_diff_{}-{}.png'.format('Real',
                    'Hybrid'), bbox_inches='tight')
        plt.close()
    else:
        import warnings
        warnings.warn('Skipping grid difference plots')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions')
    parser.add_argument('--positions', '-p', type=str,
                        help='Path to the trajectory file',
                        required=True)
    parser.add_argument('--type',
                        nargs='+',
                        default=['Real', 'Hybrid', 'Virtual'],
                        # choices=['Real', 'Hybrid', 'Virtual']
                        )
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
    parser.add_argument('--separate',
                        action='store_true',
                        help='Different grid graph for each agent',
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
        else:
            if os.path.exists(args.path + '/' + t):
                exp_files[t] = '/' + t + '/*positions.dat'

    plot(exp_files, './', args)
