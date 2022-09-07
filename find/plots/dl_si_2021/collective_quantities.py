#!/usr/bin/env python
import glob
import argparse
from turtle import position

from find.utils.features import Velocities
from find.plots.common import *
import find.plots.common as shared

import find.plots.spatial.interindividual_distance as interd
import find.plots.spatial.relative_orientation as relor

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)

ROBOT_DATA = True
TRAJNET_DATA = False
PFW_DATA = False
DISABLE_TOULOUSE = False

# TRAJNET_DATA = False
# PFW_DATA = False
# DISABLE_TOULOUSE = False

# TRAJNET_DATA = False
# PFW_DATA = True
# DISABLE_TOULOUSE = False

# TRAJNET_DATA = True
# PFW_DATA = False
# DISABLE_TOULOUSE = True


def reset_palette():
    if TRAJNET_DATA:
        shared._uni_pallete = ["#000000", "#ed8b02", "#e74c3c"]
    elif PFW_DATA:
        shared._uni_pallete = ["#000000", "#D980FA"]
    elif ROBOT_DATA:
        shared._uni_pallete = ["#000000", "#e74c3c", "#2596be"]
    else:
        shared._uni_pallete = ["#000000", "#e74c3c", "#3498db", "#2ecc71"]


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
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        data[e]['rvel'] = []
        data[e]['interindividual_distance'] = []
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
                positions = np.loadtxt(p) * args.radius
            if e == 'Robot':
                velocities = Velocities([positions], 0.1).get()[0]
            else:
                velocities = Velocities([positions], args.timestep).get()[0]
            linear_velocity = np.array((velocities.shape[0], 1))
            tup = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)

            distance = np.sqrt(
                (positions[:, 0] - positions[:, 2]) ** 2 + (positions[:, 1] - positions[:, 3]) ** 2)

            data[e]['rvel'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
            data[e]['interindividual_distance'].append(distance)

    ###############################################################################
    # Virtual
    ###############################################################################
    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.38}
                         )

    # distance to wall
    distances = {}
    for k in data.keys():
        distances[k] = data[k]['interindividual_distance']
    sub_data = distances.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']

    reset_palette()
    ax[0] = interd.interindividual_distance(sub_data, ax[0], args, [0, 35])
    yscale = 100
    ax[0] = annot_axes(ax[0],
                       r'$d_{ij}$ (cm)',
                       r'PDF $(\times {})$'.format(yscale),
                       [0.0, 25.0], [0.0, 15.0],
                       #    [0.0, 35.0], [0.0, 15.0],
                       [5, 2.5], [3, 1.5],
                       yscale)

    # relative orientation
    sub_data = data.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    reset_palette()
    relor.relative_orientation_to_neigh(sub_data, ax[1], args)
    yscale = 1000
    ax[1] = annot_axes(ax[1],
                       r'$\phi_{ij}$ $(^{\circ})$',
                       r'PDF $(\times {})$'.format(yscale),
                       [-180, 180.0], [0.0, 15.0],
                       [90, 30], [3, 1.5],
                       yscale)

    # viewing angle
    sub_data = data.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    reset_palette()
    relor.viewing_angle(sub_data, ax[2], args)
    yscale = 1000
    ax[2] = annot_axes(ax[2],
                       r'$\psi_{ij}$ $(^{\circ})$',
                       r'PDF $(\times {})$'.format(yscale),
                       [-180, 180.0], [0.0, 14.5],
                       [90, 30], [3, 1.5],
                       yscale)

    # ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
    #            fontsize=18, transform=ax[0].transAxes)
    # ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
    #            fontsize=18, transform=ax[1].transAxes)
    # ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
    #            fontsize=18, transform=ax[2].transAxes)

    plt.gcf().subplots_adjust(bottom=0.141, left=0.062, top=0.965, right=0.985)
    plt.savefig(path + 'collective_quantities_virtual.png')

    ###############################################################################
    # Hybrid
    ###############################################################################
    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.38}
                         )

    sub_data = distances.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    reset_palette()
    ax[0] = interd.interindividual_distance(sub_data, ax[0], args, [0, 30])
    yscale = 100
    ax[0] = annot_axes(ax[0],
                       r'$d_{ij}$ (cm)',
                       r'PDF $(\times {})$'.format(yscale),
                       #    [0.0, 25.0], [0.0, 15.0],
                       [0.0, 35.0], [0.0, 15.0],
                       [5, 2.5], [3, 1.5],
                       yscale)

    sub_data = data.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    reset_palette()
    relor.relative_orientation_to_neigh(sub_data, ax[1], args)
    yscale = 1000
    ax[1] = annot_axes(ax[1],
                       r'$\phi_{ij}$ $(^{\circ})$',
                       r'PDF $(\times {})$'.format(yscale),
                       [-180, 180.0], [0.0, 15.0],
                       [90, 30], [3, 1.5],
                       yscale)

    sub_data = data.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    reset_palette()
    relor.viewing_angle(sub_data, ax[2], args)
    yscale = 1000
    ax[2] = annot_axes(ax[2],
                       r'$\psi_{ij}$ $(^{\circ})$',
                       r'PDF $(\times {})$'.format(yscale),
                       [-180, 180.0], [0.0, 14.5],
                       [90, 30], [3, 1.5],
                       yscale)

    # ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
    #            fontsize=18, transform=ax[0].transAxes)
    # ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
    #            fontsize=18, transform=ax[1].transAxes)
    # ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
    #            fontsize=18, transform=ax[2].transAxes)

    plt.gcf().subplots_adjust(bottom=0.141, left=0.062, top=0.965, right=0.985)
    plt.savefig(path + 'collective_quantities_hybrid.png')
