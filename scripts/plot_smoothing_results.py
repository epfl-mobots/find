#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from pprint import pprint

import scipy.signal as signal
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot two trajectory signals against each other')
    parser.add_argument('--path1', 
                        type=str,
                        help='Path to the first trajectory file',
                        required=True)
    parser.add_argument('--path2', 
                        type=str,
                        help='Path to the second trajectory file (filtered)',
                        required=True)
    args = parser.parse_args()

    p1 = np.loadtxt(args.path1)
    p2 = np.loadtxt(args.path2)

    v1 = np.loadtxt(args.path1.replace('positions', 'velocities'))

    if os.path.exists(args.path2.replace('positions', 'velocities').replace('filtered', 'filtered_twice')):
        v2 = np.loadtxt(args.path2.replace('positions', 'velocities').replace('filtered', 'filtered_twice'))
    else:
        v2 = np.loadtxt(args.path2.replace('positions', 'velocities'))
    
    plt.plot(p1[:, 0])
    plt.plot(p2[:, 0])
    plt.show()

    plt.plot(p1[:, 1])
    plt.plot(p2[:, 1])
    plt.show()

    plt.plot(v1[:, 0] * 0.29)
    plt.plot(v2[:, 0] * 0.29)
    plt.show()

    plt.plot(v1[:, 1] * 0.29)
    plt.plot(v2[:, 1] * 0.29)
    plt.show()


