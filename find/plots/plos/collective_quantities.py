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
            positions = np.loadtxt(p) * args.radius
            velocities = Velocities([positions], args.timestep).get()[0]
            linear_velocity = np.array((velocities.shape[0], 1))
            tup = []
            dist_mat = []
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

    _, ax = plt.subplots(figsize=(10, 6),
                         nrows=2, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.38}
                         )

    # distance to wall
    distances = {}
    for k in data.keys():
        distances[k] = data[k]['interindividual_distance']

    shared._uni_pallete = ["#34495e", "#3498db"]
    sub_data = distances.copy()
    del sub_data['Hybrid']
    ax[0, 0] = interd.interindividual_distance(sub_data, ax[0, 0], args)
    ax[0, 0].set_xlabel('d (m)')
    ax[0, 0].set_ylabel('PDF')
    ax[0, 0].set_xlim([-0.02, 0.6])
    # ax[0, 0].legend()

    shared._uni_pallete = ["#9b59b6", "#34495e"]
    sub_data = distances.copy()
    del sub_data['Virtual']
    ax[1, 0] = interd.interindividual_distance(sub_data, ax[1, 0], args)
    ax[1, 0].set_xlabel('d (m)')
    ax[1, 0].set_ylabel('PDF')
    ax[1, 0].set_xlim([-0.02, 0.6])
    # ax[1, 0].legend()

    # relative orientation
    shared._uni_pallete = ["#34495e", "#3498db"]
    sub_data = data.copy()
    del sub_data['Hybrid']
    relor.relative_orientation_to_neigh(sub_data, ax[0, 1], args)
    ax[0, 1].set_xlabel(r'$\phi$ $(^{\circ})$')
    ax[0, 1].set_ylabel('PDF')
    # ax[0, 1].legend()
    ax[0, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 5))

    shared._uni_pallete = ["#9b59b6", "#34495e"]
    sub_data = data.copy()
    del sub_data['Virtual']
    relor.relative_orientation_to_neigh(sub_data, ax[1, 1], args)
    ax[1, 1].set_xlabel(r'$\phi$ $(^{\circ})$')
    ax[1, 1].set_ylabel('PDF')
    # ax[1, 1].legend()
    ax[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 5))

    # viewing angle
    shared._uni_pallete = ["#34495e", "#3498db"]
    sub_data = data.copy()
    del sub_data['Hybrid']
    relor.viewing_angle(sub_data, ax[0, 2], args)
    ax[0, 2].set_xlabel(r'$\psi$ $(^{\circ})$')
    ax[0, 2].set_ylabel('PDF')
    ax[0, 2].set_xticks(np.arange(-180, 181, 60))
    # ax[0, 2].legend()
    ax[0, 2].ticklabel_format(axis='y', style='sci', scilimits=(1, 3))

    shared._uni_pallete = ["#9b59b6", "#34495e"]
    sub_data = data.copy()
    del sub_data['Virtual']
    relor.viewing_angle(sub_data, ax[1, 2], args)
    ax[1, 2].set_xlabel(r'$\psi$ $(^{\circ})$')
    ax[1, 2].set_ylabel('PDF')
    ax[1, 2].set_xticks(np.arange(-180, 181, 60))
    # ax[1, 2].legend()
    ax[1, 2].ticklabel_format(axis='y', style='sci', scilimits=(1, 3))

    ax[0, 0].text(-0.2, 1.07, r'$\mathbf{A}$',
                  fontsize=25, transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.2, 1.07, r'$\mathbf{B}$',
                  fontsize=25, transform=ax[0, 1].transAxes)
    ax[0, 2].text(-0.2, 1.07, r'$\mathbf{C}$',
                  fontsize=25, transform=ax[0, 2].transAxes)
    ax[1, 0].text(-0.2, 1.07, r'$\mathbf{A^\prime}$',
                  fontsize=25, transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.2, 1.07, r'$\mathbf{B^\prime}$',
                  fontsize=25, transform=ax[1, 1].transAxes)
    ax[1, 2].text(-0.2, 1.07, r'$\mathbf{C^\prime}$',
                  fontsize=25, transform=ax[1, 2].transAxes)

    plt.savefig(path + 'collective_quantities.png')
