#!/usr/bin/env python
import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path

import scipy.signal as signal
import matplotlib.pyplot as plt

sys.path.append('..')
from zebra_python.features import Velocities

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

    positions = np.loadtxt(args.path)
    velocities = np.loadtxt(args.path.replace('positions', 'velocities'))

    rvelocities = []
    for i in range(velocities.shape[0]):
        r = np.sqrt(velocities[i, 1] ** 2 +
                    velocities[i, 0] ** 2 +
                    2 * velocities[i, 1] * velocities[i, 0] * np.cos(np.arctan2(velocities[i, 1], velocities[i, 0])))
        rvelocities.append(r)
    rvelocities = np.array(rvelocities)

    peaks = signal.find_peaks_cwt(rvelocities, np.arange(0.1, 0.3))
    valleys = signal.find_peaks_cwt(1 / rvelocities, np.arange(0.1, 0.3))

    print('Num peaks: ', len(peaks))

    plt.plot(rvelocities)
    plt.plot(valleys, rvelocities[valleys], 'o')
    plt.savefig(args.output + '.png', dpi=300)