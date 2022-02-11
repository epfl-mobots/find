#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
from find.plots.common import *
import find.plots.common as shared

import find.plots.spatial.interindividual_distance as interd
import find.plots.spatial.relative_orientation as relor


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

    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.38}
                         )

    # distance to wall
    distances = {}
    for k in data.keys():
        distances[k] = data[k]['interindividual_distance']

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = distances.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    ax[0] = interd.interindividual_distance(sub_data, ax[0], args, [0, 30])
    ax[0].set_xlabel('d (cm)')
    ax[0].set_xlim([0.0, 30])
    ax[0].set_xticks(np.arange(0, 31, 5))

    ax[0].set_ylabel(r'PDF $(\times 100)$')
    yticks = np.arange(0, 0.151, 0.025)
    yticklabels = ['{:.1f}'.format(e) for e in yticks * 100]
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(yticklabels)
    # ax[0].legend()

    # relative orientation
    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    relor.relative_orientation_to_neigh(sub_data, ax[1], args)
    ax[1].set_xlabel(r'$\phi$ $(^{\circ})$')
    ax[1].set_xticks(np.arange(-180, 181, 60))
    ax[1].set_xlim([-180, 180])

    ax[1].set_ylabel(r'PDF $(\times 1000)$')
    yticks =np.arange(0, 0.0151, 0.0025)
    yticklabels = ['{:.1f}'.format(e) for e in yticks * 1000]
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(yticklabels)
    # ax[1].legend()

    # # viewing angle
    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    if 'Hybrid' in sub_data.keys():
        del sub_data['Hybrid']
    relor.viewing_angle(sub_data, ax[2], args)
    ax[2].set_xlabel(r'$\psi$ $(^{\circ})$')
    ax[2].set_xticks(np.arange(-180, 181, 60))
    ax[2].set_xlim([-180, 180])

    ax[2].set_ylabel(r'PDF $(\times 1000)$')
    yticks =np.arange(0, 0.015, 0.0025)
    yticklabels = ['{:.1f}'.format(e) for e in yticks * 1000]
    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels(yticklabels)
    # ax[2].legend()

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=18, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=18, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=18, transform=ax[2].transAxes)

    plt.gcf().subplots_adjust(bottom=0.16, left=0.065, top=0.87, right=0.985)
    plt.savefig(path + 'collective_quantities_virtual.png')

    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.38}
                         )

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = distances.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    ax[0] = interd.interindividual_distance(sub_data, ax[0], args, [0, 30])
    ax[0].set_xlabel('d (cm)')
    ax[0].set_xlim([0.0, 30])
    ax[0].set_xticks(np.arange(0, 31, 5))

    ax[0].set_ylabel(r'PDF $(\times 100)$')
    yticks = np.arange(0, 0.151, 0.025)
    yticklabels = ['{:.1f}'.format(e) for e in yticks * 100]
    ax[0].set_yticks(yticks)
    ax[0].set_yticklabels(yticklabels)
    # ax[0].legend()

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    relor.relative_orientation_to_neigh(sub_data, ax[1], args)
    ax[1].set_xlabel(r'$\phi$ $(^{\circ})$')
    ax[1].set_xticks(np.arange(-180, 181, 60))
    ax[1].set_xlim([-180, 180])

    ax[1].set_ylabel(r'PDF $(\times 1000)$')
    yticks =np.arange(0, 0.0151, 0.0025)
    yticklabels = ['{:.1f}'.format(e) for e in yticks * 1000]
    ax[1].set_yticks(yticks)
    ax[1].set_yticklabels(yticklabels)
    # ax[1].legend()

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    if 'Virtual' in sub_data.keys():
        del sub_data['Virtual']
    if 'Virtual (Toulouse)' in sub_data.keys():
        del sub_data['Virtual (Toulouse)']
    relor.viewing_angle(sub_data, ax[2], args)
    ax[2].set_xlabel(r'$\psi$ $(^{\circ})$')
    ax[2].set_xticks(np.arange(-180, 181, 60))
    ax[2].set_xlim([-180, 180])

    ax[2].set_ylabel(r'PDF $(\times 1000)$')
    yticks =np.arange(0, 0.015, 0.0025)
    yticklabels = ['{:.1f}'.format(e) for e in yticks * 1000]
    ax[2].set_yticks(yticks)
    ax[2].set_yticklabels(yticklabels)
    # ax[2].legend()

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=18, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=18, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=18, transform=ax[2].transAxes)

    plt.gcf().subplots_adjust(bottom=0.16, left=0.065, top=0.87, right=0.985)
    plt.savefig(path + 'collective_quantities_hybrid.png')
