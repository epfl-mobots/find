#!/usr/bin/env python

import os
import glob
import argparse
import numpy as np

from tqdm import tqdm, trange
from find.plots.common import *


from pathlib import Path
import matplotlib
from scipy.interpolate import griddata
import scipy.stats as st


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


def plot_grid_differences(files, path, args):
    keys = args.type
    if 'Real' not in keys and ('Hybrid' not in keys or 'Virtual' not in keys):
        import warnings
        warnings.warn('Skipping grid difference plots')
        return

    keys.remove('Real')
    skipped = 0

    desc = 'Occupancy grid distribution difference (Skipped {})'
    num_files = trange(
        len(files), desc=desc.format(skipped), leave=True)

    for i in num_files:
        f = files[i]
        r_xx = np.loadtxt(f.replace('f_Real', 'xx_Real'))
        r_yy = np.loadtxt(f.replace('f_Real', 'yy_Real'))
        r_f = np.loadtxt(f)

        for k in keys:
            if not os.path.isfile(f.replace('_Real', '_{}'.format(k))):
                skipped += 1
                num_files.set_description(desc.format(skipped), refresh=True)
                continue
            comp_f = np.loadtxt(f.replace('_Real', '_{}'.format(k)))
            comp_f[comp_f <= 1e-40] = 0
            r_f[r_f <= 1e-40] = 0
            f_diff = r_f - comp_f

            cmap = matplotlib.cm.get_cmap('jet')
            _ = plt.figure(figsize=(6, 6))
            ax = plt.gca()

            ax.contourf(r_xx, r_yy, np.abs(f_diff), levels=100, cmap=cmap)
            outer = plt.Circle(
                (0, 0), 0.25, color='k', fill=False)
            ax.add_artist(outer)
            ax.set_xlim([-0.3, 0.3])
            ax.set_ylim([-0.3, 0.3])

            plt.savefig(
                '/' + '/'.join(f.split('/')[:-1]) + '/occupancy_dist_diff_{}-{}.png'.format('Real', k))
            print('/'.join(f.split('/')
                  [:-1]) + '/occupancy_dist_diff_{}-{}.png'.format('Real', k))
            input()
            plt.close()


def plot(exp_files, path, args):
    files = []
    for p in Path(path).rglob('f_Real.dat'):
        files.append(str(p))
    plot_grid_differences(files, path, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize grid position differences')
    parser.add_argument('--type',
                        nargs='+',
                        default=['Real', 'Hybrid', 'Virtual'],
                        choices=['Real', 'Hybrid', 'Virtual'])
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
