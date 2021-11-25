#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
from find.plots.common import *
import find.plots.common as shared

import find.plots.spatial.grid_occupancy as go


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e] = []
        for p in pos:
            positions = np.loadtxt(p) * args.radius
            data[e].append(positions)

    fig, ax = plt.subplots(figsize=(10, 4),
                           nrows=1, ncols=3,
                           gridspec_kw={'width_ratios': [
                               1, 1, 1], 'wspace': 0.45}
                           )

    ax[0] = go.occupancy_grid(data, fig, 'Real', ax[0], args, 0.2)
    ax[1] = go.occupancy_grid(data, fig, 'Hybrid', ax[1], args, 0.2)
    ax[2] = go.occupancy_grid(data, fig, 'Virtual', ax[2], args, 0.2)

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=25, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=25, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=25, transform=ax[2].transAxes)
    ax[0].set_ylabel('y (m)')
    ax[0].set_xlabel('x (m)')
    ax[1].set_ylabel('y (m)')
    ax[1].set_xlabel('x (m)')
    ax[2].set_ylabel('y (m)')
    ax[2].set_xlabel('x (m)')
    ax[0].grid(linestyle='dotted')
    ax[1].grid(linestyle='dotted')
    ax[2].grid(linestyle='dotted')

    plt.savefig(path + 'occupancy_maps.png')

    fig, ax = plt.subplots(figsize=(10, 4),
                           nrows=1, ncols=3,
                           gridspec_kw={'width_ratios': [
                               1, 1, 1], 'wspace': 0.45}
                           )

    ax[0] = go.occupancy_grid_dist(data, fig, 'Real', ax[0], args, 0.2)
    ax[1] = go.occupancy_grid_dist(data, fig, 'Hybrid', ax[1], args, 0.2)
    ax[2] = go.occupancy_grid_dist(data, fig, 'Virtual', ax[2], args, 0.2)

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=25, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=25, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=25, transform=ax[2].transAxes)
    ax[0].set_ylabel('y (m)')
    ax[0].set_xlabel('x (m)')
    ax[1].set_ylabel('y (m)')
    ax[1].set_xlabel('x (m)')
    ax[2].set_ylabel('y (m)')
    ax[2].set_xlabel('x (m)')
    ax[0].grid(linestyle='dotted')
    ax[1].grid(linestyle='dotted')
    ax[2].grid(linestyle='dotted')

    plt.savefig(path + 'occupancy_maps_pdfs.png')
