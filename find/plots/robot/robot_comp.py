#!/usr/bin/env python
import os
import glob
import argparse
# from turtle import left, position
from scipy import stats

from find.utils.features import Velocities, Accelerations
from find.plots.common import *
import find.plots.common as shared

import find.plots.spatial.resultant_velocity as rv
import find.plots.spatial.resultant_acceleration as ra
import find.plots.spatial.distance_to_wall as dtw
import find.plots.spatial.relative_orientation as relor

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)


def reset_palette():
    shared._uni_pallete = ['#1e81b0', '#48A14D', '#e67e22']


def annot_axes(ax, xlabel, ylabel, xlim, ylim, xloc, yloc, yscale):
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MultipleLocator(xloc[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(xloc[1]))
    ax.tick_params(axis="x", which='both',
                   direction="in")
    ax.tick_params(axis="x", which='major', length=4.0, width=0.7)
    ax.tick_params(axis="x", which='minor', length=2.0, width=0.7)

    ax.set_ylabel(ylabel)
    ylim = [e / yscale for e in ylim]
    ax.set_ylim(ylim)
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: '{:.1f}'.format(x * yscale, ',')))
    ax.yaxis.set_major_locator(MultipleLocator(yloc[0] / yscale))
    ax.yaxis.set_minor_locator(MultipleLocator(yloc[1] / yscale))
    ax.tick_params(which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.grid(False)
    return ax


def plot(exp_files, path, args):
    data = {}
    data['path'] = path

    for e in sorted(exp_files.keys()):
        if e == 'BOBI':
            timestep = args.bt
        elif e == 'F44' or 'FishBot' in e:
            timestep = args.f44t
        else:
            timestep = args.timestep

        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        data[e]['acc'] = []
        data[e]['rvel'] = []
        data[e]['distance_to_wall'] = []
        if args.robot:
            data[e]['ridx'] = []
        sample_count = 0
        for p in pos:
            if e == 'Virtual (Toulouse)' and not DISABLE_TOULOUSE:
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
                positions = np.loadtxt(p)
                positions = np.loadtxt(p) * args.radius

            num_inds = positions.shape[1] // 2
            sample_count += positions.shape[0]

            print('Using timestep {} for {}'.format(timestep, e))
            velocities = Velocities([positions], timestep).get()[0]
            accelerations = Accelerations(
                [velocities], timestep).get()[0]

            tup = []
            dist_mat = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)

                distance = args.radius - \
                    np.sqrt(positions[:, i * 2] ** 2 +
                            positions[:, i * 2 + 1] ** 2)
                dist_mat.append(distance)
            dist_mat = np.array(dist_mat).T

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                if os.path.exists(r):
                    ridx = np.loadtxt(r).astype(int)
                else:
                    ridx = -1
                data[e]['ridx'].append(int(ridx))

            data[e]['rvel'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
            data[e]['distance_to_wall'].append(dist_mat)

            tup = []
            for i in range(accelerations.shape[1] // 2):
                linear_acc = np.sqrt(
                    accelerations[:, i * 2] ** 2 + accelerations[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_acc)
            data[e]['acc'].append(np.array(tup).T)

        print('Samples for {}: {}'.format(e, sample_count))

    ###############################################################################
    # plotting
    ###############################################################################
    _, axs = plt.subplots(figsize=(10, 10),
                          nrows=2, ncols=2,
                          gridspec_kw={
        'width_ratios': [1, 1],
        'height_ratios': [1, 1],
        'wspace': 0.2, 'hspace': 0.2}
    )
    ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]

    # velocity
    sub_data = data.copy()

    reset_palette()
    ax[0] = rv.compute_resultant_velocity(sub_data, ax[0], args, [0, 41])
    yscale = 100
    ax[0] = annot_axes(ax[0],
                       '$V$ (cm/s)', r'PDF $(\times {})$'.format(yscale),
                       #    [0.0, 35.0], [0.0, 7.2],
                       #    [0.0, 35.0], [0.0, 22],
                       #    [0.0, 35.0], [0.0, 9],
                       [0.0, 35.0], [0.0, 15],
                       [5, 2.5], [2, 1],
                       yscale)

    # accel
    sub_data = data.copy()

    # reset_palette()
    ax[1] = ra.compute_resultant_acceleration(sub_data, ax[1], args, [0, 175])
    yscale = 100
    ax[1] = annot_axes(ax[1],
                       r'$\alpha$ ($cm/s^2$)', r'PDF $(\times {})$'.format(yscale),
                       #    [0.0, 35.0], [0.0, 7.2],
                       #    [0.0, 35.0], [0.0, 22],
                       #    [0.0, 35.0], [0.0, 9],
                       [0.0, 120.0], [0.0, 6.0],
                       [35, 7], [1, 0.2],
                       yscale)

    # distance to wall
    sub_data = data.copy()

    reset_palette()
    dtw.distance_plot(sub_data, ax[2], args, [0, 25])
    yscale = 100
    ax[2] = annot_axes(ax[2],
                       r'$r_w$ (cm)', r'PDF $(\times {})$'.format(yscale),
                       [0.0, 10.0], [0.0, 60],
                       [2, 0.4], [10, 2],
                       yscale)

    # relative angle to the wall
    sub_data = data.copy()

    reset_palette()
    relor.relative_orientation_to_wall(sub_data, ax[3], args)
    yscale = 100
    ax[3] = annot_axes(ax[3],
                       r'$\theta_{\rm w}$ $(^{\circ})$',
                       r'PDF $(\times {})$'.format(yscale),
                       [0, 180], [0, 7.5],
                       [90, 30], [1.5, 0.3],
                       yscale)

    # ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
    #            fontsize=18, transform=ax[0].transAxes)
    # ax[2].text(-0.2, 1.07, r'$\mathbf{B}$',
    #            fontsize=18, transform=ax[2].transAxes)
    # ax[3].text(-0.2, 1.07, r'$\mathbf{C}$',
    #            fontsize=18, transform=ax[3].transAxes)

    plt.gcf().subplots_adjust(bottom=0.141, left=0.057, top=0.965, right=0.985)
    plt.savefig(path + 'robot_comp.png', dpi=600)
