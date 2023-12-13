#!/usr/bin/env python
import os
import glob
import argparse

from find.utils.utils import angle_to_pipi, compute_leadership
from find.utils.features import Velocities
from find.plots.common import *


def plot(exp_files, path, args):
    data = np.loadtxt(args.path + '/simu_data.txt')

    fig = plt.figure()
    ax = plt.gca()

    data1 = np.sort(data[:, 0])
    data2 = np.sort(data[:, 1])
    data3 = np.sort(data[:, 2])

    markers1 = data1[data1 < 6000.0]
    markers2 = data2[data2 < 6000.0]
    markers3 = data3[data3 < 6000.0]
    data1_plot = [0] + list(markers1) + [6000]
    data2_plot = [0] + list(markers2) + [6000]
    data3_plot = [0] + list(markers3) + [6000]

    y = 1 - np.arange(0, len(data1_plot)) / data.shape[0]

    plt.plot(data1_plot, y, label='Threshold = 1 m',
             linewidth=2.0, color='C0')
    plt.plot(data2_plot, y, label='Threshold = 3 m',
             linewidth=2.0, color='C1')
    plt.plot(data3_plot, y, label='Threshold = 10 m',
             linewidth=2.0, color='C2')

    ymarkers1 = 1 - np.arange(1, markers1.shape[0]+1) / data1.shape[0]
    ymarkers2 = 1 - np.arange(1, markers2.shape[0]+1) / data2.shape[0]
    ymarkers3 = 1 - np.arange(1, markers3.shape[0]+1) / data3.shape[0]
    plt.plot(markers1, ymarkers1, linestyle='None',
             marker='o', markersize=3.5, color='C0')
    plt.plot(markers2, ymarkers2, linestyle='None',
             marker='o', markersize=3.5, color='C1')
    plt.plot(markers3, ymarkers3, linestyle='None',
             marker='o', markersize=3.5, color='C2')

    mean1 = np.mean(markers1)
    median1 = np.median(markers1)
    mean2 = np.mean(markers2)
    median2 = np.median(markers2)
    mean3 = np.mean(markers3)
    median3 = np.median(markers3)
    print('Means: {} {} {}'.format(mean1, mean2, mean3))
    print('Median: {} {} {}'.format(median1, median2, median3))

    plt.axvline(mean1, 0, 1.0, color='C0', linestyle=':',
                linewidth=1.5)
    plt.axvline(mean2, 0, 1.0, color='C1', linestyle=':',
                linewidth=1.5)
    plt.axvline(mean3, 0, 1.0, color='C2', linestyle=':',
                linewidth=1.5)

    plt.xlim([0, 6000])
    plt.ylim([0, 1])
    plt.xlabel('Escape time (s)', fontsize=16)
    plt.ylabel('Survival probability', fontsize=16)
    ax.legend()

    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)

    plt.tight_layout()
    plt.savefig(path + 'simu_comp.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Distance to wall figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--radius', '-r', type=float,
                        help='Raidus',
                        default=0.25,
                        required=False)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--kde_gridsize',
                        type=int,
                        help='Grid size for kernel density estimation plots',
                        default=1500,
                        required=False)
    parser.add_argument('--type',
                        nargs='+',
                        default=['Real', 'Hybrid', 'Virtual'],
                        choices=['Real', 'Hybrid', 'Virtual'])
    parser.add_argument('--original_files',
                        type=str,
                        default='raw/*processed_positions.dat',
                        required=False)
    parser.add_argument('--hybrid_files',
                        type=str,
                        default='generated/*generated_positions.dat',
                        required=False)
    parser.add_argument('--virtual_files',
                        type=str,
                        default='generated/*generated_virtu_positions.dat',
                        required=False)
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
