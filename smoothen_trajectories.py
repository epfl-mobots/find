#!/usr/bin/env python
import glob
import argparse
import numpy as np
from pathlib import Path
from pykalman import KalmanFilter

from features import Velocities


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess fish trajectories')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--fps', type=int,
                        help='Camera framerate',
                        required=True)
    parser.add_argument('--centroids', '-c', type=int,
                        help='Frames to use in order to compute the centroidal positions',
                        required=True)
    args = parser.parse_args()

    timestep = args.centroids / args.fps

    files = glob.glob(args.path + '/*processed_positions.dat')
    for f in files:
        positions = np.loadtxt(f)

        mus = []
        for i in range(positions.shape[1] // 2):
            kf = KalmanFilter(n_dim_state=2, n_dim_obs=2)
            mu, sigma = kf.filter(positions[:, i*2:i*2+2])
            if len(mus) == 0:
                mus = np.array(mu)
            else:
                mus = np.append(mus, np.array(mu))
        velocities = Velocities([mus], timestep).get()

        new_f = f.replace('positions.dat', 'positions_filtered.dat', 1)
        np.savetxt(new_f, mus)
        new_f = f.replace('positions.dat', 'velocities_filtered.dat', 1)
        np.savetxt(new_f, velocities[0])

