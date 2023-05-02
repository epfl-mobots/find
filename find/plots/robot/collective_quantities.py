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

ROBOT_DATA = False
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
        shared._uni_pallete = ["#000000", "#e74c3c"]
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
    data['path'] = path

    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        data[e]['rvel'] = []
        data[e]['interindividual_distance'] = []
        if args.robot:
            data[e]['ridx'] = []
            data[e]['ci'] = []
            data[e]['d1'] = []
            data[e]['pi'] = []

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
            num_inds = positions.shape[1] // 2

            if e == 'BOBI' or 'Simu' in e:
                velocities = Velocities([positions], args.bt).get()[0]
                print('Using timestep {} for {}'.format(args.bt, e))
            else:
                print('Using timestep {} for {}'.format(args.timestep, e))
                velocities = Velocities([positions], args.timestep).get()[0]

            linear_velocity = np.array((velocities.shape[0], 1))
            hdgs = np.empty((positions.shape[0], 0))
            tup = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)

                hdg = np.arctan2(velocities[:, i*2+1], velocities[:, i*2])
                hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

            if num_inds > 2:
                # Ci
                dist = []
                for i in range(num_inds):
                    dist_i = np.zeros((positions.shape[0],))
                    for j in range(num_inds):
                        if i == j:
                            continue
                        dist_i += (positions[:, i*2] - positions[:, j*2]) ** 2 + \
                            (positions[:, i*2+1] - positions[:, j*2+1]) ** 2
                    cdist_i = np.sqrt(dist_i / (num_inds - 1))
                    dist.append(cdist_i.tolist())
                data[e]['ci'].append(dist)

                # d1
                ndists = []
                for idx in range(num_inds):
                    ndist = np.zeros((positions.shape[0],))
                    for row in range(positions.shape[0]):
                        distances = []
                        for nidx in range(num_inds):
                            if idx == nidx:
                                continue
                            eucd = np.sqrt((positions[row, idx*2] - positions[row, nidx*2]) ** 2 +
                                           (positions[row, idx*2 + 1] - positions[row, nidx*2 + 1]) ** 2)
                            distances.append(eucd)
                        ndist[row] = np.min(distances)
                    ndists.append(ndist.tolist())
                data[e]['d1'].append(ndists)

                # Pi
                dist = []
                for i in range(num_inds):
                    dist_i = np.zeros((positions.shape[0],))
                    for j in range(num_inds):
                        if i == j:
                            continue
                        dist_i += np.cos(hdgs[:, i] - hdgs[:, j])
                    cdist_i = dist_i / (num_inds - 1)
                    dist.append(cdist_i.tolist())
                data[e]['pi'].append(dist)
            else:
                distance = np.sqrt(
                    (positions[:, 0] - positions[:, 2]) ** 2 + (positions[:, 1] - positions[:, 3]) ** 2)
                data[e]['interindividual_distance'].append(distance.tolist())

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

            data[e]['rvel'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

    ###############################################################################
    # plotting
    ###############################################################################
    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.38}
                         )

    # interindividual
    reset_palette()
    yscale = 100
    if num_inds == 2:
        distances = {}
        for k in data.keys():
            if k == 'path':
                continue
            distances[k] = data[k]['interindividual_distance']
        distances['path'] = data['path']
        sub_data = distances.copy()

        ax[0] = interd.interindividual_distance(sub_data, ax[0], args, [0, 50])
        ax[0] = annot_axes(ax[0],
                           r'$d_{ij}$ (cm)',
                           r'PDF $(\times {})$'.format(yscale),
                           [0.0, 25.0], [0.0, 15.0],
                           #    [0.0, 35.0], [0.0, 15.0],
                           [5, 2.5], [3, 1.5],
                           yscale)
    elif num_inds == 5:
        ax[0] = interd.ci(
            data, 'ci', ax[0], args, [0, 50])
        ax[0] = annot_axes(ax[0],
                           r'$P(C_{i})$ (cm)',
                           r'PDF $(\times {})$'.format(yscale),
                           [0.0, 30.0], [0.0, 18.2],
                           #    [0.0, 35.0], [0.0, 15.0],
                           [5, 2.5], [3, 1.5],
                           yscale)

    # 2nd panel: rel orientation or nearest neigh for N > 2
    sub_data = data.copy()
    reset_palette()
    if num_inds == 2:
        yscale = 1000
        ax[1] = relor.relative_orientation_to_neigh(sub_data, ax[1], args)

        ax[1] = annot_axes(ax[1],
                           r'$\phi_{ij}$ $(^{\circ})$',
                           r'PDF $(\times {})$'.format(yscale),
                           [0, 180.0], [0.0, 30.0],
                           [90, 30], [5, 2.5],
                           yscale)
    elif num_inds == 5:
        yscale = 100
        ax[1] = interd.ci(data, 'd1', ax[1], args, [0, 50])

        ax[1] = annot_axes(ax[1],
                           r'$P(d_{1})$ (cm)',
                           r'PDF $(\times {})$'.format(yscale),
                           [0.0, 30.0], [0.0, 30],
                           #    [0.0, 35.0], [0.0, 15.0],
                           [5, 2.5], [3, 1.5],
                           yscale)

    # 3rd panel: viewing angle or cos dif(hdg) N > 2
    sub_data = data.copy()
    reset_palette()
    if num_inds == 2:
        yscale = 1000
        relor.viewing_angle(sub_data, ax[2], args)
        ax[2] = annot_axes(ax[2],
                           r'$\psi_{ij}$ $(^{\circ})$',
                           r'PDF $(\times {})$'.format(yscale),
                           [-180, 180.0], [0.0, 6.5],
                           [90, 30], [3, 1.5],
                           yscale)
    elif num_inds == 5:
        yscale = 1
        ax[2] = interd.ci(data, 'pi', ax[2], args, [-1, 1])
        ax[2] = annot_axes(ax[2],
                           r'$P(P_{i})$',
                           r'PDF',
                           [0.25, 1], [0.0, 16],
                           [0.25, 0.05], [4, 0.8],
                           yscale)

    # ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
    #            fontsize=18, transform=ax[0].transAxes)
    # ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
    #            fontsize=18, transform=ax[1].transAxes)
    # ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
    #            fontsize=18, transform=ax[2].transAxes)

    plt.gcf().subplots_adjust(bottom=0.141, left=0.062, top=0.965, right=0.985)
    plt.savefig(path + 'collective_quantities.png')
