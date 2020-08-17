#!/usr/bin/env python

import matplotlib

matplotlib.use('Agg')

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

plt.style.use('dark_background')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions')
    parser.add_argument('--positions', '-p', type=str,
                        help='Path to the trajectory file',
                        required=True)
    parser.add_argument('--fname', '-o', type=str,
                        help='output file name',
                        required=True)
    parser.add_argument('--open', action='store_true',
                        help='Visualize the open setup', default=False)
    parser.add_argument('--regex', action='store_true',
                        help='Flag to signify that args.positions is a regex',
                        default=False)

    args = parser.parse_args()

    iradius = 0.655172413793
    oradius = 1.0
    center = (0, 0)
    radius = (iradius, oradius)

    fig = plt.figure(figsize=(6, 7))
    ax = plt.gca()

    inner = plt.Circle(
        center, radius[0], color='white', fill=False)
    outer = plt.Circle(
        center, radius[1], color='white', fill=False)
    # if not args.open:
    #     ax.add_artist(inner)
    # ax.add_artist(outer)

    bins = 120
    y, x = np.meshgrid(np.linspace(center[0] - (oradius + 0.0001),
                                   center[0] + (oradius + 0.0001), bins),
                       np.linspace(center[1] - (oradius + 0.0001),
                                   center[1] + (oradius + 0.0001), bins))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    outside_els = np.sum(r > radius[1])

    z = np.zeros([bins, bins])


    traj_list = []
    if not args.regex:
        traj_list.append(np.loadtxt(args.positions))
    else:
        files = glob.glob(args.positions)
        for f in files:
            traj_list.append(np.loadtxt(f))

    total_steps = 0
    for traj in traj_list:
        tsteps = traj.shape[0]
        total_steps += tsteps
        individuals = traj.shape[1] // 2
        idcs = range(individuals)

        for i in range(tsteps):
            for j in idcs:
                traj_x = traj[i, j * 2]
                traj_y = traj[i, j * 2 + 1]

                dist_x = np.abs(np.array(traj_x - x[:, 0]))
                dist_y = np.abs(np.array(traj_y - y[0, :]))
                min_xidx = np.argmin(dist_x)
                min_yidx = np.argmin(dist_y)
                z[min_xidx, min_yidx] += 1
    z /= total_steps
    # z *= 100
    # print(outside_els)
    # z /= (np.size(z) - outside_els)
    z_min, z_max = 0, 0.0011

    print(np.max(z))

    palette = sns.color_palette('RdYlBu_r', 1000)
    palette = [(0, 0, 0, 0)] + palette
    sns.set_palette(palette)
    palette = sns.color_palette()
    cmap = ListedColormap(palette.as_hex())

    c = ax.pcolormesh(x, y, z, cmap=cmap, vmin=z_min, vmax=z_max)
    fig.colorbar(c, ax=ax, label='Cell occupancy (%)', orientation='horizontal', pad=0.05)

    # ax.axis('off')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    plt.tight_layout()
    plt.savefig(args.fname + '.png', dpi=300)
