#!/usr/bin/env python
import glob
import string
import argparse
import itertools

from find.utils.features import Velocities
from find.plots.common import *
import find.plots.common as shared

import find.plots.spatial.grid_occupancy as go

abc = list(string.ascii_uppercase)
abc_it = iter(abc)


def plot(exp_files, path, args):
    data = {}
    grids = {}
    ridcs = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e] = []
        if args.robot:
            ridcs[e] = []

        for p in pos:
            if e == 'Virtual (Toulouse)':
                f = open(p)
                # to allow for loading fortran's doubles
                strarray = f.read().replace("D+", "E+").replace("D-", "E-")
                f.close()
                num_ind = len(strarray.split('\n')[0].strip().split('  '))
                positions = np.fromstring(
                    strarray, sep='\n').reshape(-1, num_ind) * args.radius
            else:
                positions = np.loadtxt(p) * args.radius
            data[e].append(positions)
            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                ridcs[e].append(int(ridx))

        if args.separate:
            if args.robot:
                x, y, z = go.construct_grid_sep(
                    data, e, args, sigma=1, ridcs=ridcs)
            else:
                x, y, z = go.construct_grid_sep(data, e, args, sigma=1)
        else:
            x, y, z = go.construct_grid(data, e, args, sigma=1)
        grid = {'x': x, 'y': y, 'z': z}
        grids[e] = grid

    orig_cutoff = args.grid_cutoff_val
    # This was set to match the max among all grids for H. Rhodostomus
    # args.grid_cutoff_val = 0.004

    if len(data.keys()) > 1:

        if args.separate:
            num_exps = len(data.keys())

            fig = plt.figure()
            fig.set_figwidth(5)
            fig.set_figheight(9)

            gs = fig.add_gridspec(num_exps + 1, 1, hspace=0.35, wspace=0.2)
            gs_cbar = gs[-1].subgridspec(1, 1)

            cmesh = None
            for ne, e in enumerate(data.keys()):
                num_inds = data[e][0].shape[1] // 2

                order = list(range(num_inds))
                if args.robot:
                    ridx = ridcs[e][ne]
                    if ridx >= 0:
                        order.remove(ridx)
                        order = [ridx] + order

                sep_grids = grids[e]
                gsrow = gs[ne].subgridspec(
                    1, num_inds, wspace=0.0, hspace=-0.2)
                for i, idx in enumerate(order):
                    ax = fig.add_subplot(gsrow[0, i])
                    g = {
                        'x': sep_grids['x'][idx],
                        'y': sep_grids['y'][idx],
                        'z': sep_grids['z'][idx]
                    }

                    title = 'Individual {}'.format(idx)
                    if args.robot and i == 0 and ridx > 0:
                        title = 'Robot'
                    # title = ''

                    ax, cmesh = go.occupancy_grid(data, g,
                                                  fig, title, ax,
                                                  args, pad=0.0, draw_colorbar=False, draw_circle=False)
                    # ax.set_ylabel('y (m)')
                    # ax.set_xlabel('x (m)')
                    ax.grid(linestyle='dotted')
                    # if i > 0:
                    ax.get_yaxis().set_ticklabels([])
                    ax.get_xaxis().set_ticklabels([])

            ax_cbar = fig.add_subplot(gs_cbar[0])
            cbar = fig.colorbar(cmesh, ax=ax_cbar, label='Cell occupancy (%)',
                                location='top', extend='max', fraction=0.08, pad=0.3)
            cbar.ax.tick_params(rotation=30)
            ax_cbar.get_xaxis().set_ticks([])
            ax_cbar.get_yaxis().set_ticks([])
            ax_cbar.axis('off')
            ax_cbar.set_visible(False)

            plt.tight_layout()
            plt.savefig(path + 'occupancy_maps_sep.png', bbox_inches='tight')

            args.grid_cutoff_val = orig_cutoff  # resetting this

        else:
            combs = list(itertools.combinations(
                list(data.keys()), 2))

            fig = plt.figure()
            fig.set_figwidth(9)
            fig.set_figheight(7)

            gs = fig.add_gridspec(len(combs) + 1, 1)
            # gs.set_height_ratios([2.75, 1])
            gs_cbar = gs[-1].subgridspec(1, 1)

            cmesh = None
            for nc, c in enumerate(combs):
                gsrow = gs[nc].subgridspec(1, 3, wspace=0.25)
                ax0 = fig.add_subplot(gsrow[0, 0])
                ax1 = fig.add_subplot(gsrow[0, 1])
                ax2 = fig.add_subplot(gsrow[0, 2])

                ax0, cmesh = go.occupancy_grid(data, grids[c[1]],
                                               fig, c[1], ax0,
                                               args, pad=0.25, draw_colorbar=False, draw_circle=False)

                ax1, cmesh = go.occupancy_grid(data, grids[c[0]],
                                               fig, c[0], ax1,
                                               args, pad=0.25, draw_colorbar=False, draw_circle=False)
                ax2, _ = go.grid_difference(grids,
                                            c[1], c[0],
                                            fig, ax2, args, draw_colorbar=False, draw_circle=False)

                lbl = r'$\mathbf{' + next(abc_it) + '}$'
                ax0.text(-0.2, 1.07, lbl,
                         fontsize=25, transform=ax0.transAxes)

                lbl = r'$\mathbf{' + next(abc_it) + '}$'
                ax1.text(-0.2, 1.07, lbl,
                         fontsize=25, transform=ax1.transAxes)
                lbl = r'$\mathbf{' + next(abc_it) + '}$'
                ax2.text(-0.2, 1.07, lbl,
                         fontsize=25, transform=ax2.transAxes)

                ax0.set_ylabel('y (m)')
                ax0.set_xlabel('x (m)')
                ax1.set_ylabel('y (m)')
                ax1.set_xlabel('x (m)')
                ax2.set_ylabel('y (m)')
                ax2.set_xlabel('x (m)')
                # ax1.get_xaxis().set_ticklabels([])
                ax1.get_yaxis().set_ticklabels([])
                # ax2.get_xaxis().set_ticklabels([])
                ax2.get_yaxis().set_ticklabels([])
                ax0.grid(linestyle='dotted')
                ax1.grid(linestyle='dotted')
                ax2.grid(linestyle='dotted')

            ax_cbar = fig.add_subplot(gs_cbar[0])
            cbar = fig.colorbar(cmesh, ax=ax_cbar, label='Cell occupancy (%)',
                                location='top', extend='max', pad=0.3)
            cbar.ax.tick_params(rotation=30)
            ax_cbar.get_xaxis().set_ticks([])
            ax_cbar.get_yaxis().set_ticks([])
            ax_cbar.axis('off')
            ax_cbar.set_visible(False)

            plt.tight_layout()
            plt.savefig(path + 'occupancy_maps.png', bbox_inches='tight')

            args.grid_cutoff_val = orig_cutoff  # resetting this
