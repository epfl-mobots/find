#!/usr/bin/env python
import glob
import argparse
from turtle import position

from find.utils.features import Velocities
from find.plots.common import *
import find.plots.common as shared
from find.utils.utils import angle_to_pipi

from find.plots.correlation.position_correlation import corx
from find.plots.correlation.velocity_correlation import corv
from find.plots.correlation.relative_orientation_correlation import cortheta

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)


TRAJNET_DATA = False
PFW_DATA = False
DISABLE_TOULOUSE = False

# TRAJNET_DATA = False
# PFW_DATA = True
# DISABLE_TOULOUSE = False

# TRAJNET_DATA = True
# PFW_DATA = False
# DISABLE_TOULOUSE = True


def reset_palette():
    # shared._uni_pallete = ["#000000", "#e74c3c", "#3498db", "#2ecc71"]
    if TRAJNET_DATA:
        shared._uni_pallete = ["#000000", "#ed8b02", "#e74c3c"]
    elif PFW_DATA:
        shared._uni_pallete = ["#000000", "#D980FA"]
    else:
        shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]


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
        data[e]['rel_or'] = []
        for p in pos:
            if e == 'Virtual (Toulouse)' and not DISABLE_TOULOUSE:
                f = open(p)
                # to allow for loading fortran's doubles
                strarray = f.read().replace("D+", "E+").replace("D-", "E-")
                f.close()
                num_ind = len(strarray.split('\n')[0].strip().split('  '))
                positions = np.fromstring(
                    strarray, sep='\n').reshape(-1, num_ind) * args.radius
            else:
                positions = np.loadtxt(p) * args.radius
            if args.num_virtual_samples > 0:
                positions = positions[:args.num_virtual_samples]
            velocities = Velocities([positions], args.timestep).get()[0]
            linear_velocity = np.array((velocities.shape[0], 1))
            tup = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)

            hdgs = np.empty((positions.shape[0], 0))
            for i in range(positions.shape[1] // 2):
                hdg = np.arctan2(velocities[:, i*2+1], velocities[:, i*2])
                hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

            # for the focal
            angle_dif_focal = hdgs[:, 0] - \
                np.arctan2(positions[:, 1], positions[:, 0])
            angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))

            # for the neigh
            angle_dif_neigh = hdgs[:, 1] - \
                np.arctan2(positions[:, 3], positions[:, 2])
            angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))

            data[e]['rel_or'].append(
                np.array([angle_dif_focal, angle_dif_neigh]).T)

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
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.0}
                         )

    # position

    sub_data = data.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    reset_palette()
    ax[0] = corx(sub_data, ax[0], args)
    ax[0] = annot_axes(ax[0],
                       '$t$ (s)', r'$C_X$ $(cm^2)$',
                       [0.0, 25.0], [0.0, 1300],
                       [5, 2.5], [250, 125],
                       1)
    print('Done with position')

    # velocity
    sub_data = data.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    reset_palette()
    ax[1] = corv(sub_data, ax[1], args)
    ax[1] = annot_axes(ax[1],
                       '$t$ (s)', r'$C_V$ $(\,cm^2 / \,s^2)$',
                       [0.0, 25.0], [-100.0, 200],
                       [5, 2.5], [50, 25],
                       1)
    ax[1].yaxis.set_label_coords(-0.18, 0.5)
    print('Done with Velocity')

    # relative orientation
    sub_data = data.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    reset_palette()
    ax[2] = cortheta(sub_data, ax[2], args)
    ax[2] = annot_axes(ax[2],
                       '$t$ (s)', r'$C_{\theta_{\rm w}}$',
                       [0.0, 25.0], [0.0, 1.0],
                       [5, 2.5], [0.2, 0.1],
                       1)
    print('Done with theta')

    # ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
    #            fontsize=18, transform=ax[0].transAxes)
    # ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
    #            fontsize=18, transform=ax[1].transAxes)
    # ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
    #            fontsize=18, transform=ax[2].transAxes)

    ax[0].legend().remove()
    ax[1].legend().remove()
    ax[2].legend().remove()

    plt.gcf().subplots_adjust(bottom=0.141, left=0.078, top=0.965, right=0.985)
    plt.savefig(path + 'correlation_quantities_virtual.png')

    exit(1)

    ###############################################################################
    # Hybrid
    ###############################################################################
    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.0}
                         )

    sub_data = data.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    reset_palette()
    ax[0] = corx(sub_data, ax[0], args)
    ax[0] = annot_axes(ax[0],
                       '$t$ (s)', r'$C_X$ $(cm^2)$',
                       [0.0, 25.0], [0.0, 1300],
                       [5, 2.5], [250, 125],
                       1)
    print('Done with position')

    sub_data = data.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    reset_palette()
    ax[1] = corv(sub_data, ax[1], args)
    ax[1] = annot_axes(ax[1],
                       '$t$ (s)', r'$C_V$ $(\,cm^2 / \,s^2)$',
                       [0.0, 25.0], [-100.0, 200],
                       [5, 2.5], [50, 25],
                       1)
    ax[1].yaxis.set_label_coords(-0.18, 0.5)
    print('Done with Velocity')

    shared._uni_pallete = ["#e74c3c", "#000000", "#3498db"]
    sub_data = data.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    reset_palette()
    ax[2] = cortheta(sub_data, ax[2], args)
    ax[2] = annot_axes(ax[2],
                       '$t$ (s)', r'$C_{\theta_{\rm w}}$',
                       [0.0, 25.0], [0.0, 1.0],
                       [5, 2.5], [0.2, 0.1],
                       1)
    print('Done with theta')

    # ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
    #            fontsize=18, transform=ax[0].transAxes)
    # ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
    #            fontsize=18, transform=ax[1].transAxes)
    # ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
    #            fontsize=18, transform=ax[2].transAxes)

    ax[0].legend().remove()
    ax[1].legend().remove()
    ax[2].legend().remove()

    plt.gcf().subplots_adjust(bottom=0.135, left=0.078, top=0.965, right=0.985)
    plt.savefig(path + 'correlation_quantities_hybrid.png')

    print('Done with relative orientation to the wall')
