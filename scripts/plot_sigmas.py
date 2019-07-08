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
        description='Preprocess fish trajectories')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the simgas file',
                        required=True)
    args = parser.parse_args()

    sigmas = np.loadtxt(args.path)

    plt.figure()
    plt.plot(sigmas[:, 0])
    plt.savefig('varx.png', dpi=300)

    plt.figure()
    plt.plot(sigmas[:, 1])
    plt.savefig('vary.png', dpi=300)
