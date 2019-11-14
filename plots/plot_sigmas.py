#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import numpy as np

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
