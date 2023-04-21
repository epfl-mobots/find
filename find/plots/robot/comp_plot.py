#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities, Accelerations
from find.utils.utils import compute_leadership
from find.plots.common import *

from find.plots.robot.velocity_bplots import vel_plots
from find.plots.robot.acceleration_bplots import acc_plots
from find.plots.robot.interindividual_dist_bplots import idist_plots, cdist_plots
from find.plots.robot.occupancy_grids import grid_plot
from find.plots.robot.activity_bplots import activity_plots
import find.plots.spatial.grid_occupancy as go

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)


def comp_plot_single(data, grids, path, args):
    palette = ['#1e81b0', '#D61A3C', '#48A14D']

    fig = plt.figure()

    fig.set_figwidth(9)
    fig.set_figheight(5)
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 3], wspace=0.05)

    # -- main grid row
    gcol0 = gs[0].subgridspec(1, 1, wspace=0.0, hspace=0.0)
    gcol1 = gs[1].subgridspec(
        1, 2, wspace=0.25, hspace=0.0, width_ratios=[2, 2])

    # grids
    ridcs = {}
    sdata = {}
    for e in data.keys():
        ridcs[e] = data[e]['ridx']
        sdata[e] = data[e]['pos']
    ogs = gcol0[0].subgridspec(
        1, 2, hspace=0.0, wspace=0.0,
        # height_ratios=[3, 3, 1]
        width_ratios=[1, 10]
    )
    ogs = grid_plot(sdata, grids, path, fig, ogs, args, ridcs=ridcs)

    # velocity
    gv = gcol1[0, 0].subgridspec(len(data.keys()), 1, wspace=0.0, hspace=0.5)

    ax = [fig.add_subplot(gv[i, 0]) for i in range(len(data.keys()))]
    ax = vel_plots(data, path, ax, args, orient='h', palette=palette)
    for cax in ax:
        cax.set_ylim([-0.5, 0.5])
        cax.set_xlim([0, 40])
        cax.xaxis.set_major_locator(MultipleLocator(5))
        cax.xaxis.set_minor_locator(MultipleLocator(1))
        cax.tick_params(axis='x', which='both', bottom=True,
                        left=True, right=True, top=True)
        cax.tick_params(axis="x", which='both', direction="in")
        cax.set_yticklabels([])
        cax.set_title('')
        cax.tick_params(axis='both', which='major', labelsize=11)
    # for cax in ax[:-1]:
    #     cax.set_xticklabels([])

    # acceleration
    ga = gcol1[0, 1].subgridspec(len(data.keys()), 1, wspace=0.0, hspace=0.5)
    ax = [fig.add_subplot(ga[i, 0]) for i in range(len(data.keys()))]

    ax = acc_plots(data, path, ax, args, orient='h', palette=palette)
    for cax in ax:
        cax.set_ylim([-0.5, 0.5])
        cax.set_xlim([0, 125])
        cax.xaxis.set_major_locator(MultipleLocator(25))
        cax.xaxis.set_minor_locator(MultipleLocator(5))
        cax.tick_params(axis='x', which='both', bottom=True,
                        left=True, right=True, top=True)
        cax.tick_params(axis="x", which='both', direction="in")
        cax.set_yticklabels([])
        cax.set_title('')
        cax.tick_params(axis='both', which='major', labelsize=11)
    # for cax in ax[:-1]:
    #     cax.set_xticklabels([])

    # activity
    # gac = gcol1[0, 3].subgridspec(1, 1, wspace=0.0, hspace=0.0)
    # ax = fig.add_subplot(gi[1, 0])
    # activity_plots(data, path, ax, args, freezing=True, palette=palette)
    # ax.set_xticklabels(list(data.keys()))
    # ax.set_ylim([0, 100])
    # ax.yaxis.set_major_locator(MultipleLocator(25))
    # ax.yaxis.set_minor_locator(MultipleLocator(5))
    # ax.tick_params(axis='y', which='both', bottom=True,
    #                left=True, right=True, top=True)
    # ax.tick_params(axis="y", which='both', direction="in")
    # ax.set_ylabel('Inactivity %', fontsize=11)
    # # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    # ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()

    plt.tight_layout()
    plt.savefig(path + 'comp_plot.tiff', bbox_inches='tight')


def comp_plot_pair(data, grids, path, args):
    palette = ['#1e81b0', '#D61A3C', '#48A14D']

    fig = plt.figure()

    if len(data.keys()) == 3:
        fig.set_figwidth(13)
        fig.set_figheight(5)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 3], wspace=0.05)
    else:
        fig.set_figwidth(15)
        fig.set_figheight(5)
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 3], wspace=0.05)

    # -- main grid row
    gcol0 = gs[0].subgridspec(1, 1, wspace=0.0, hspace=0.0)

    if len(data.keys()) == 3:
        gcol1 = gs[1].subgridspec(
            1, 3, wspace=0.25, hspace=0.0, width_ratios=[2, 2, 1.5])
    else:
        gcol1 = gs[1].subgridspec(
            1, 3, wspace=0.25, hspace=0.0, width_ratios=[2, 2, 1])

    # grids
    ridcs = {}
    sdata = {}
    for e in data.keys():
        ridcs[e] = data[e]['ridx']
        sdata[e] = data[e]['pos']
    ogs = gcol0[0].subgridspec(
        1, 2, hspace=0.0, wspace=0.0,
        # height_ratios=[3, 3, 1]
        width_ratios=[1, 10]
    )
    ogs = grid_plot(sdata, grids, path, fig, ogs, args, ridcs=ridcs)

    # velocity
    gv = gcol1[0, 0].subgridspec(len(data.keys()), 1, wspace=0.0, hspace=0.5)

    ax = [fig.add_subplot(gv[i, 0]) for i in range(len(data.keys()))]
    ax = vel_plots(data, path, ax, args, orient='h', palette=palette)
    for cax in ax:
        if 'Fish' in data.keys():
            cax.set_xlim([0, 30])
        else:
            cax.set_xlim([0, 25])
        cax.set_ylim([-0.5, 1.5])
        cax.xaxis.set_major_locator(MultipleLocator(5))
        cax.xaxis.set_minor_locator(MultipleLocator(1))
        cax.tick_params(axis='x', which='both', bottom=True,
                        left=True, right=True, top=True)
        cax.tick_params(axis="x", which='both', direction="in")
        cax.set_yticklabels([])
        cax.set_title('')
        cax.tick_params(axis='both', which='major', labelsize=11)
    # for cax in ax[:-1]:
    #     cax.set_xticklabels([])

    # acceleration
    ga = gcol1[0, 1].subgridspec(len(data.keys()), 1, wspace=0.0, hspace=0.5)
    ax = [fig.add_subplot(ga[i, 0]) for i in range(len(data.keys()))]

    ax = acc_plots(data, path, ax, args, orient='h', palette=palette)
    for cax in ax:
        cax.set_xlim([0, 75])
        cax.set_ylim([-0.5, 1.5])
        cax.xaxis.set_major_locator(MultipleLocator(25))
        cax.xaxis.set_minor_locator(MultipleLocator(5))
        cax.tick_params(axis='x', which='both', bottom=True,
                        left=True, right=True, top=True)
        cax.tick_params(axis="x", which='both', direction="in")
        cax.set_yticklabels([])
        cax.set_title('')
        cax.tick_params(axis='both', which='major', labelsize=11)
    # for cax in ax[:-1]:
    #     cax.set_xticklabels([])

    # interindividual
    gi = gcol1[0, 2].subgridspec(2, 1, wspace=0.0, hspace=0.25)

    ax = fig.add_subplot(gi[0, 0])
    ax = idist_plots(data, path, ax, args, orient='h', palette=palette)
    if 'circle' in path:
        ax.set_xlim([0, 50])
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    elif 'eights' in path:
        ax.set_xlim([0, 25])
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        ax.set_xlim([0, 25])
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.tick_params(axis='x', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="x", which='both', direction="in")
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()

    # activity
    # gac = gcol1[0, 3].subgridspec(1, 1, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gi[1, 0])
    activity_plots(data, path, ax, args, freezing=True,
                   orient='h', palette=palette)
    ax.set_yticklabels(list(data.keys()))
    ax.set_xlim([0, 100])
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='x', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="x", which='both', direction="in")
    ax.set_xlabel('Inactivity %', fontsize=11)
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()

    plt.tight_layout()
    plt.savefig(path + 'comp_plot.tiff', bbox_inches='tight')


def comp_plot_five(data, grids, path, args):
    palette = ['#1e81b0', '#48A14D']

    fig = plt.figure()
    fig.set_figwidth(16)
    fig.set_figheight(5)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3], wspace=0.05)

    # -- main grid row
    gcol0 = gs[0].subgridspec(1, 1, wspace=0.0, hspace=0.0)

    gcol1 = gs[1].subgridspec(
        1, 3, wspace=0.25, hspace=0.0, width_ratios=[2, 2, 1.5])

    # grids
    ridcs = {}
    sdata = {}
    for e in data.keys():
        ridcs[e] = data[e]['ridx']
        sdata[e] = data[e]['pos']
    ogs = gcol0[0].subgridspec(
        1, 2, hspace=0.0, wspace=0.0,
        # height_ratios=[3, 3, 1]
        width_ratios=[1, 10]
    )
    ogs = grid_plot(sdata, grids, path, fig, ogs, args, ridcs=ridcs)

    # velocity
    gv = gcol1[0, 0].subgridspec(len(data.keys()), 1, wspace=0.0, hspace=0.5)

    ax = [fig.add_subplot(gv[i, 0]) for i in range(len(data.keys()))]
    ax = vel_plots(data, path, ax, args, orient='h', palette=palette)
    for cax in ax:
        cax.set_ylim([-0.5, 4.5])
        cax.set_xlim([0, 40])
        cax.xaxis.set_major_locator(MultipleLocator(5))
        cax.xaxis.set_minor_locator(MultipleLocator(1))
        cax.tick_params(axis='x', which='both', bottom=True,
                        left=True, right=True, top=True)
        cax.tick_params(axis="x", which='both', direction="in")
        cax.set_yticklabels([])
        cax.set_title('')
        cax.tick_params(axis='both', which='major', labelsize=11)
    # for cax in ax[:-1]:
    #     cax.set_xticklabels([])

    # acceleration
    ga = gcol1[0, 1].subgridspec(len(data.keys()), 1, wspace=0.0, hspace=0.5)
    ax = [fig.add_subplot(ga[i, 0]) for i in range(len(data.keys()))]

    ax = acc_plots(data, path, ax, args, orient='h', palette=palette)
    for cax in ax:
        cax.set_xlim([0, 125])
        cax.xaxis.set_major_locator(MultipleLocator(25))
        cax.xaxis.set_minor_locator(MultipleLocator(5))
        cax.tick_params(axis='x', which='both', bottom=True,
                        left=True, right=True, top=True)
        cax.tick_params(axis="x", which='both', direction="in")
        cax.set_yticklabels([])
        cax.set_title('')
        cax.tick_params(axis='both', which='major', labelsize=11)
    # for cax in ax[:-1]:
    #     cax.set_xticklabels([])

    # # interindividual
    # gi = gcol1[0, 2].subgridspec(2, 1, wspace=0.0, hspace=0.5)

    # ax = [fig.add_subplot(gi[1, 0]), fig.add_subplot(gi[0, 0])]
    # ax = cdist_plots(data, path, ax, args, orient='h', palette=palette)

    # for idx, cax in enumerate(ax):
    #     cax.set_xlim([0, 30.0])
    #     cax.xaxis.set_major_locator(MultipleLocator(5.0))
    #     cax.xaxis.set_minor_locator(MultipleLocator(1))
    #     cax.tick_params(axis='x', which='both', bottom=True,
    #                     left=True, right=True, top=True)
    #     cax.tick_params(axis="x", which='both', direction="in")
    #     # cax.set_yticklabels([])
    #     cax.set_yticklabels([])
    #     cax.set_title('')
    #     cax.tick_params(axis='both', which='major', labelsize=11)
    #     # cax.yaxis.set_label_position("right")
    #     # cax.yaxis.tick_right()
    #     # if 'Biomimetic' in list(data.keys())[idx]:
    #     #     cax.set_yticklabels(
    #     #         ['Biomimetic\nlure', 'Fish 0 to 3\n(average)'], rotation=90)
    #     # else:
    #     #     cax.set_yticklabels(
    #     #         ['Fish 0', 'Fish 1 to 4\n(average)'], rotation=90)

    # interindividual
    gi = gcol1[0, 2].subgridspec(2, 1, wspace=0.0, hspace=0.5)

    ax = fig.add_subplot(gi[0, 0])
    ax = idist_plots(data, path, ax, args, orient='h', palette=palette)
    ax.set_xlim([0, 25.0])

    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.tick_params(axis='x', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="x", which='both', direction="in")
    ax.set_yticklabels([])
    # ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()

    # activity
    # gac = gcol1[0, 3].subgridspec(1, 1, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gi[1, 0])
    activity_plots(data, path, ax, args, orient='h',
                   freezing=True, palette=palette)
    ax.set_yticklabels(list(data.keys()))
    ax.set_xlim([0, 100])
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='x', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="x", which='both', direction="in")
    ax.set_xlabel('Inactivity %', fontsize=11)
    ax.set_yticklabels([])
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()

    plt.tight_layout()
    plt.savefig(path + 'comp_plot.tiff', bbox_inches='tight')


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
        data[e]['rvel'] = []
        data[e]['racc'] = []
        data[e]['idist'] = []
        data[e]['cdist'] = []
        data[e]['samples'] = np.loadtxt(
            args.path + '/' + e + '/sample_counts.txt')
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
            velocities = Velocities([positions], timestep).get()[0][1:-1]
            accelerations = Accelerations(
                [velocities], timestep).get()[0][1:-1]

            samples += positions.shape[0]
            num_inds = positions.shape[1] // 2

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

            tup = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)
            mat = np.array(tup).T
            mat = mat[np.all(mat <= 42, axis=1), :]
            data[e]['rvel'].append(mat)

            tup = []
            for i in range(accelerations.shape[1] // 2):
                linear_acceleration = np.sqrt(
                    accelerations[:, i * 2] ** 2 + accelerations[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_acceleration)
            mat = np.array(tup).T
            mat = mat[np.all(mat <= 175, axis=1), :]
            data[e]['racc'].append(mat)

            interind_dist = []
            if num_inds >= 2:
                dist = np.zeros((1, positions.shape[0]))
                for i in range(1, num_inds):
                    dist += (positions[:, 0] - positions[:, i*2]) ** 2 + \
                        (positions[:, 1] - positions[:, i*2 + 1]) ** 2
                interind_dist = np.sqrt(dist / (num_inds - 1))
                interind_dist = interind_dist.tolist()[0]
            # elif num_inds == 2:
            #     interind_dist = np.sqrt(
            #         (positions[:, 0] - positions[:, 2]) ** 2 + (positions[:, 1] - positions[:, 3]) ** 2).tolist()
            else:
                print('Single fish. Skipping interindividual distance')

            data[e]['idist'].append(interind_dist)

            if num_inds == 5:
                dist = np.zeros((num_inds,))
                for idx in range(num_inds):
                    for row in range(positions.shape[0]):
                        distances = []
                        for nidx in range(num_inds):
                            if idx == nidx:
                                continue
                            eucd = np.sqrt((positions[row, idx*2] - positions[row, nidx*2]) ** 2 +
                                           (positions[row, idx*2 + 1] - positions[row, nidx*2 + 1]) ** 2)
                            distances.append(eucd)
                        mdist = np.min(distances)
                        dist[idx] += mdist

                cdist = dist / positions.shape[0]
                cdist = cdist.tolist()
                if args.robot:
                    reduced_list = [cdist[ridx]]
                    neighs = []
                    for i, v in enumerate(cdist):
                        if i == ridx:
                            continue
                        neighs.append(v)
                    reduced_list.append(np.mean(neighs))
                else:
                    reduced_list = [cdist[0], np.mean(cdist[1:])]
                data[e]['cdist'].append(reduced_list)

            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
        print('{} has {} samples'.format(e, samples))

        # construct grids
        ridcs = {}
        ridcs[e] = data[e]['ridx']
        sdata = {}
        sdata[e] = data[e]['pos']

        x, y, z = go.construct_grid_sep(
            sdata, e, args, sigma=1, ridcs=ridcs)
        grid = {'x': x, 'y': y, 'z': z}
        grids[e] = grid

    if num_inds == 1:
        comp_plot_single(data, grids, path, args)
    elif num_inds == 2:
        comp_plot_pair(data, grids, path, args)
    elif num_inds == 5:
        comp_plot_five(data, grids, path, args)
    else:
        print(num_inds)
        assert False, 'Unsupported number of individuals'
