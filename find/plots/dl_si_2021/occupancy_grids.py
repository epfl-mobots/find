#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
from find.plots.common import *
import find.plots.common as shared

import find.plots.spatial.grid_occupancy as go


def plot(exp_files, path, args):
    data = {}
    grids = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e] = []
        for p in pos:
            positions = np.loadtxt(p) * args.radius
            data[e].append(positions)

        x, y, z = go.construct_grid(data, e, args)
        grid = {'x': x, 'y': y, 'z': z}
        grids[e] = grid

    fig = plt.figure()
    fig.set_figwidth(9)
    fig.set_figheight(7)
    gs = fig.add_gridspec(2, 1)
    gs.set_height_ratios([2.75, 1])

    gs_cbar = gs[1].subgridspec(1, 1)

    gs0 = gs[0].subgridspec(1, 2)
    gs00 = gs0[0].subgridspec(1, 1)
    gs01 = gs0[1].subgridspec(2, 2, hspace=0.15)

    ax0 = fig.add_subplot(gs00[0])
    ax1 = fig.add_subplot(gs01[0, 0])
    ax2 = fig.add_subplot(gs01[0, 1])
    ax3 = fig.add_subplot(gs01[1, 0])
    ax4 = fig.add_subplot(gs01[1, 1])
    ax_cbar = fig.add_subplot(gs_cbar[0])

    orig_cutoff = args.grid_cutoff_val
    # This was set to match the max among all grids for H. Rhodostomus
    args.grid_cutoff_val = 0.002

    ax0, cmesh = go.occupancy_grid(data, grids['Real'],
                                   fig, 'Real', ax0,
                                   args, pad=0.25, draw_colorbar=False)
    ax1, _ = go.occupancy_grid(data, grids['Virtual'],
                               fig, 'Virtual', ax1, args, draw_colorbar=False)
    ax2, _ = go.grid_difference(grids,
                                'Real', 'Virtual',
                                fig, ax2, args, draw_colorbar=False)

    ax3, _ = go.occupancy_grid(data, grids['Hybrid'],
                               fig, 'Hybrid', ax3, args, draw_colorbar=False)
    ax4, _ = go.grid_difference(grids,
                                'Real', 'Hybrid',
                                fig, ax4, args, draw_colorbar=False)

    cbar = fig.colorbar(cmesh, ax=ax_cbar, label='Cell occupancy (%)',
                        location='top', extend='max', pad=0.3)
    cbar.ax.tick_params(rotation=30)

    ax0.text(-0.1, 1.07, r'$\mathbf{A}$',
             fontsize=25, transform=ax0.transAxes)
    ax1.text(-0.1, 1.07, r'$\mathbf{B}$',
             fontsize=25, transform=ax1.transAxes)
    ax2.text(-0.1, 1.07, r'$\mathbf{C}$',
             fontsize=25, transform=ax2.transAxes)
    ax3.text(-0.1, 1.07, r'$\mathbf{D}$',
             fontsize=25, transform=ax3.transAxes)
    ax4.text(-0.1, 1.07, r'$\mathbf{E}$',
             fontsize=25, transform=ax4.transAxes)

    ax0.set_ylabel('y (m)')
    ax0.set_xlabel('x (m)')
    ax1.get_xaxis().set_ticklabels([])
    ax1.get_yaxis().set_ticklabels([])
    ax2.get_xaxis().set_ticklabels([])
    ax2.get_yaxis().set_ticklabels([])
    ax3.get_xaxis().set_ticklabels([])
    ax3.get_yaxis().set_ticklabels([])
    ax4.get_xaxis().set_ticklabels([])
    ax4.get_yaxis().set_ticklabels([])
    ax_cbar.get_xaxis().set_ticks([])
    ax_cbar.get_yaxis().set_ticks([])
    ax_cbar.axis('off')
    ax_cbar.set_visible(False)

    ax0.grid(linestyle='dotted')
    ax1.grid(linestyle='dotted')
    ax2.grid(linestyle='dotted')
    ax3.grid(linestyle='dotted')
    ax4.grid(linestyle='dotted')

    plt.tight_layout()
    plt.savefig(path + 'occupancy_maps.png', bbox_inches='tight')

    args.grid_cutoff_val = orig_cutoff  # resetting this
