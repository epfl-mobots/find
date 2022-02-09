#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
from find.plots.common import *
import find.plots.common as shared

import find.plots.spatial.resultant_velocity as rv
import find.plots.spatial.distance_to_wall as dtw
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
        data[e]['distance_to_wall'] = []
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

            data[e]['rvel'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
            data[e]['distance_to_wall'].append(dist_mat)

    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.25, 'hspace': 0.38}
                         )

    # velocity
    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Hybrid']
    ax[0] = rv.compute_resultant_velocity(sub_data, ax[0], args)
    ax[0].set_xlabel('$V$ (m/s)')
    ax[0].set_ylabel('PDF')
    ax[0].set_xlim([-0.02, 0.6])
    ax[0].set_yticks(np.arange(0, 6.9, 2))
    ax[0].set_xticks(np.arange(0, 0.61, 0.2))
    # ax[0].legend()

    # distance to wall
    distances = {}
    positions = {}
    for k in data.keys():
        distances[k] = data[k]['distance_to_wall']
        positions[k] = data[k]['pos']

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data_d = distances.copy()
    del sub_data_d['Hybrid']
    sub_data_p = positions.copy()
    del sub_data_p['Hybrid']
    dtw.distance_plot(sub_data_d, sub_data_p, ax[1], args)
    ax[1].set_xlabel(r'$r$ (m)')
    ax[1].set_ylabel('PDF')
    ax[1].set_yticks(np.arange(0, 21, 5))
    ax[1].set_xticks(np.arange(0, 0.31, 0.1))
    # ax[1].legend()

    # relative angle to the wall
    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Hybrid']
    relor.relative_orientation_to_wall(sub_data, ax[2], args)
    ax[2].set_xlabel(r'$\theta$ $(^{\circ})$')
    ax[2].set_ylabel('PDF')
    ax[2].set_xticks(np.arange(-180, 181, 60))
    ax[2].set_yticks(np.arange(0, 0.021, 0.005))
    # ax[ 2].legend()
    ax[2].ticklabel_format(axis='y', style='sci', scilimits=(1, 3))

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=18, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=18, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=18, transform=ax[2].transAxes)
    plt.gcf().subplots_adjust(bottom=0.16, left=0.055, top=0.87, right=0.985)
    plt.savefig(path + 'individual_quantities_virtual.png')

    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.25, 'hspace': 0.38}
                         )

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Virtual']
    del sub_data['Virtual (Toulouse)']
    ax[0] = rv.compute_resultant_velocity(sub_data, ax[0], args)
    ax[0].set_xlabel('$V$ (m/s)')
    ax[0].set_ylabel('PDF')
    ax[0].set_xlim([-0.02, 0.6])
    ax[0].set_yticks(np.arange(0, 6.9, 2))
    ax[0].set_xticks(np.arange(0, 0.61, 0.2))
    # ax[0].legend()

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data_d = distances.copy()
    del sub_data_d['Virtual']
    del sub_data_d['Virtual (Toulouse)']
    sub_data_p = positions.copy()
    del sub_data_p['Virtual']
    del sub_data_p['Virtual (Toulouse)']
    dtw.distance_plot(sub_data_d, sub_data_p, ax[1], args)
    ax[1].set_xlabel(r'$r$ (m)')
    ax[1].set_ylabel('PDF')
    ax[1].set_yticks(np.arange(0, 21, 5))
    ax[1].set_xticks(np.arange(0, 0.31, 0.1))
    # ax[1].legend()

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Virtual']
    del sub_data['Virtual (Toulouse)']
    relor.relative_orientation_to_wall(sub_data, ax[2], args)
    ax[2].set_xlabel(r'$\theta$ $(^{\circ})$')
    ax[2].set_ylabel('PDF')
    ax[2].set_xticks(np.arange(-180, 181, 60))
    ax[2].set_yticks(np.arange(0, 0.021, 0.005))
    # ax[2].legend()
    ax[2].ticklabel_format(axis='y', style='sci', scilimits=(1, 3))

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=18, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=18, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=18, transform=ax[2].transAxes)

    plt.gcf().subplots_adjust(bottom=0.16, left=0.055, top=0.87, right=0.985)
    plt.savefig(path + 'individual_quantities_hybrid.png')
