#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np
from pathlib import Path
from pprint import pprint

import scipy.signal as signal
import matplotlib.pyplot as plt

from features import Velocities
from utils import ExperimentInfo, Center, Normalize



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess fish trajectories')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    args = parser.parse_args()

    files = glob.glob(args.path + '/*processed_positions_filtered.dat')
    
    all_features = np.empty(shape=[0, 3])
    for f in files:
        positions = np.loadtxt(f)
        velocities = np.loadtxt(f.replace('positions', 'velocities'))

        rvelocities = []
        for i in range(velocities.shape[0]):
            r = np.sqrt(velocities[i, 1] ** 2 +
                    velocities[i, 0] ** 2 +
                    2 * velocities[i, 1] * velocities[i, 0] * np.cos(np.arctan2(velocities[i, 1], velocities[i, 0])))
            rvelocities.append(r)
        rvelocities = np.array(rvelocities)

        # peaks = signal.find_peaks_cwt(rvelocities, np.arange(0.1,0.3))
        valleys = signal.find_peaks_cwt(1/rvelocities, np.arange(0.1,0.3))
        
        events = []
        features = []
        for i in range(len(valleys)-1):
            event = [i, np.argmax(rvelocities[valleys[i]:valleys[i+1]+1]), valleys[i], valleys[i+1]]
            events.append(event)

            peak_vel = rvelocities[event[1]]
            length = valleys[i+1] - valleys[i] + 1
            mean = np.mean(rvelocities[valleys[i]:valleys[i+1]+1])
            features.append([peak_vel, length, mean])
        all_features = np.append(all_features, np.array(features), axis=0)
        np.savetxt(f.replace('positions', 'kicks'), events)
        np.savetxt(f.replace('positions', 'kicks_features'), features)
        
    np.savetxt(args.path + '/all_kick_features.dat', all_features)     
    for n in range(all_features.shape[1]):
        all_features[:, n] = (all_features[:, n] - np.mean(all_features[:, n])) / np.std(all_features[:, n])
    np.savetxt(args.path + '/standardized_kick_features.dat', all_features) 

