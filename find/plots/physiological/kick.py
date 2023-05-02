#!/usr/bin/env python
import os
import sys
import glob
import argparse
# from turtle import position
import numpy as np
from pathlib import Path

import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

from find.utils.features import Velocities


def plot(exp_files, path, args):
    data = {}

    fig = plt.figure(figsize=(6, 3))
    ax = plt.gca()

    for e in sorted(exp_files.keys()):
        print('----- {}'.format(e))
        pl = []
        vl = []
        samples = 0

        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        data[e]['rvel'] = []

        if e == 'BOBI':
            timestep = args.bt
        elif e == 'F44':
            timestep = args.f44t
        else:
            timestep = args.timestep

        for p in pos:
            # if samples > 143000:
            #     continue
            positions = np.loadtxt(p) * args.radius
            if e == 'BOBI':
                velocities = Velocities([positions], args.bt).get()[0]
            if e == 'F44':
                velocities = Velocities([positions], args.f44t).get()[0]
            else:
                velocities = Velocities([positions], args.timestep).get()[0]

            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

            rvel = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)
            data[e]['rvel'].append(rvel)

            samples += positions.shape[0]

            # start = 4000
            # step = 50
            # if len(rvel) < start + step:
            #     continue
            # rvel = rvel[start: (start + step)]

            coef = 2
            xticks = np.arange(0, len(rvel), coef)
            ax.set_xticks(xticks)

            xtickslabels = []
            for i in range(len(xticks)):
                xtickslabels.append('{:.2f}'.format(coef * i * timestep))
            ax.set_title(e)
            ax.set_xticklabels(xtickslabels,)

            peaks = signal.find_peaks_cwt(rvel, np.arange(0.1, 0.3))
            valleys = signal.find_peaks_cwt(
                1 / rvel, np.arange(0.1, 0.3))

            for i in range(1, len(peaks)):
                pl.append((peaks[i] - peaks[i-1]) * timestep)

            for i in range(1, len(valleys)):
                vl.append((valleys[i] - valleys[i-1]) * timestep)

            ax.set_xticks(np.arange(0, len(rvel) + 1, 25))
            # ax.plot(rvel, linewidth=0.3)
            # ax.plot(valleys, rvel[valleys], 'o', markersize=0.4)
            # break

        print('Using timestep: {}'.format(timestep))

        print('Num samples: ', samples)
        print('Num peaks: ', len(pl))

        mean = np.sum(pl) / len(pl)
        print('p2p {}'.format(mean))

        mean = np.sum(vl) / len(vl)
        print('v2v {}'.format(mean))

    # plt.xticks(rotation=90)
    # plt.savefig(path + 'kick-comp.png'.format(e), dpi=300)
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess fish trajectories')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--output', '-o', type=str,
                        help='Filename for the output file',
                        required=True)
    args = parser.parse_args()

    exp_files = {}
    for t in args.type:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
