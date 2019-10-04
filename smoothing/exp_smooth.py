#!/usr/bin/env python
import glob
import argparse
import numpy as np
from pathlib import Path
from pykalman import KalmanFilter

from features import Velocities
from utils import ExperimentInfo, Center, Normalize


def exp_filter(signal, alpha):
    filtered_signal = []
    for n in range(signal.shape[1]):
        sig = signal[:, n]
        filtered_sig = [sig[0]]
        # y(k) = y(k-1) + (1-a)*( x(k) - y(k-1) )
        for m in range(1, sig.shape[0]):
            filtered_sig.append(filtered_sig[-1] + (1-alpha) * (sig[m] - filtered_sig[-1]))
        filtered_signal.append(filtered_sig)
    return np.transpose(np.array(filtered_signal))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Exponential smoothing for the fish trajectories')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--fps', type=int,
                        help='Camera framerate',
                        required=True)
    parser.add_argument('--centroids', '-c', type=int,
                        help='Frames to use in order to compute the centroidal positions',
                        required=True)
    parser.add_argument('--alpha', '-a', type=float,
                        default=0.1, 
                        help='Smoothing factor',
                        required=False)
    parser.add_argument('--alpha_velocity', type=float,
                        default=0.1, 
                        help='Smoothing factor',
                        required=False)
    parser.add_argument('--center', action='store_true',
                        help='Center smoothed data')
    parser.add_argument('--norm', action='store_true',
                        help='Normalize smoothed data')
    args = parser.parse_args()


    timestep = args.centroids / args.fps

    files = glob.glob(args.path + '/*processed_positions.dat')
    data = []
    for f in files:
        positions = np.loadtxt(f)
        data.append(exp_filter(positions, args.alpha))

    info = ExperimentInfo(data)
    if args.center:
        data, info = Center(data, info).get()
    if args.norm:
        data, info = Normalize(data, info).get()
    velocities = Velocities(data, timestep).get()

    for i, f in enumerate(files):
        f = files[i]
        new_f = f.replace('positions.dat', 'positions_filtered.dat', 1)
        np.savetxt(new_f, data[i])
        new_f = f.replace('positions.dat', 'velocities_filtered.dat', 1)
        np.savetxt(new_f, velocities[i])
        new_f = f.replace('positions.dat', 'velocities_filtered_twice.dat', 1)
        np.savetxt(new_f, exp_filter(velocities[i], args.alpha_velocity))

