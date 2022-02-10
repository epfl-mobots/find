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

    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.3, 'hspace': 0.0}
                         )

    # position
    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Hybrid']
    ax[0] = corx(sub_data, ax[0], args)
    ax[0].set_xlabel('$t$ (s)')
    ax[0].set_ylabel(r'$C_X$ $(cm^2)$')
    ax[0].set_yticks(np.arange(0, 1251, 250))
    ax[0].set_xticks(np.arange(0, 26, 2.5))
    ax[0].set_xlim([0, 25])
    ax[0].set_ylim([0, 1250])
    ax[0].tick_params(axis='x', labelrotation=45)
    # ax[0].legend()
    print('Done with position')

    # velocity
    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Hybrid']
    ax[1] = corv(sub_data, ax[1], args)
    ax[1].set_xlabel('$t$ (s)')
    ax[1].set_ylabel(r'$C_V$ $(\,cm^2 / \,s^2)$')
    ax[1].set_yticks(np.arange(-100, 201, 50))
    ax[1].set_ylim([-101, 200])
    ax[1].set_xticks(np.arange(0, 26, 2.5))
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_xlim([0, 25])
    # ax[1].legend()
    print('Done with Velocity')

    # relative orientation
    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Hybrid']
    ax[2] = cortheta(sub_data, ax[2], args)
    ax[2].set_xlabel('$t$ (s)')
    ax[2].set_ylabel(r'$C_\theta$')
    ax[2].set_xlim([0, 25])
    ax[2].set_yticks(np.arange(-0.2, 1.01, 0.2))
    ax[2].set_xticks(np.arange(0, 26, 2.5))
    ax[2].tick_params(axis='x', labelrotation=45)
    ax[2].set_ylim([0, 1.0])
    # ax[2].legend()

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=18, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=18, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=18, transform=ax[2].transAxes)

    ax[0].legend().remove()
    ax[1].legend().remove()
    ax[2].legend().remove()

    plt.gcf().subplots_adjust(bottom=0.22, left=0.07, top=0.85, right=0.985)
    plt.savefig(path + 'correlation_quantities_virtual.png')

    _, ax = plt.subplots(figsize=(10, 3),
                         nrows=1, ncols=3,
                         gridspec_kw={'width_ratios': [
                             1, 1, 1], 'wspace': 0.25, 'hspace': 0.0}
                         )

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Virtual']
    ax[0] = corx(sub_data, ax[0], args)
    ax[0].set_xlabel('$t$ (s)')
    ax[0].set_ylabel(r'$C_X$ $(cm^2)$')
    ax[0].set_yticks(np.arange(0, 1251, 250))
    ax[0].set_xticks(np.arange(0, 26, 2.5))
    ax[0].set_xlim([0, 25])
    ax[0].set_ylim([0, 1250])
    ax[0].tick_params(axis='x', labelrotation=45)
    # ax[0].legend()
    print('Done with position')

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Virtual']
    ax[1] = corv(sub_data, ax[1], args)
    ax[1].set_xlabel('$t$ (s)')
    ax[1].set_ylabel(r'$C_V$ $(\,cm^2 / \,s^2)$')
    ax[1].set_yticks(np.arange(-100, 201, 50))
    ax[1].set_ylim([-101, 200])
    ax[1].set_xticks(np.arange(0, 26, 2.5))
    ax[1].tick_params(axis='x', labelrotation=45)
    ax[1].set_xlim([0, 25])
    # ax[1].legend()
    print('Done with Velocity')

    shared._uni_pallete = ["#000000", "#e74c3c", "#3498db"]
    sub_data = data.copy()
    del sub_data['Virtual']
    ax[2] = cortheta(sub_data, ax[2], args)
    ax[2].set_xlabel('$t$ (s)')
    ax[2].set_ylabel(r'$C_\theta$')
    ax[2].set_xlim([0, 25])
    ax[2].set_yticks(np.arange(-0.2, 1.01, 0.2))
    ax[2].set_xticks(np.arange(0, 26, 2.5))
    ax[2].tick_params(axis='x', labelrotation=45)
    ax[2].set_ylim([0, 1.0])
    # ax[2].legend()

    ax[0].text(-0.2, 1.07, r'$\mathbf{A}$',
               fontsize=18, transform=ax[0].transAxes)
    ax[1].text(-0.2, 1.07, r'$\mathbf{B}$',
               fontsize=18, transform=ax[1].transAxes)
    ax[2].text(-0.2, 1.07, r'$\mathbf{C}$',
               fontsize=18, transform=ax[2].transAxes)

    ax[0].legend().remove()
    ax[1].legend().remove()
    ax[2].legend().remove()

    plt.gcf().subplots_adjust(bottom=0.22, left=0.07, top=0.85, right=0.985)
    plt.savefig(path + 'correlation_quantities_hybrid.png')

    print('Done with relative orientation to the wall')
