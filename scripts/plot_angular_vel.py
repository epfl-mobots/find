#!/usr/bin/env python
import os
import glob
import sys
import math
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from pprint import pprint

from pylab import *
from matplotlib import gridspec
import matplotlib.lines as mlines
from cycler import cycler

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
# palette = flatui
# palette = 'Paired'
# palette = 'husl'
# palette = 'Set2'
palette = sns.cubehelix_palette(11)
colors = sns.color_palette(palette)
sns.set(style="darkgrid")


gfontsize = 10
params = {
    'axes.labelsize': gfontsize,
    'font.size': gfontsize,
    'legend.fontsize': gfontsize,
    'xtick.labelsize': gfontsize,
    'ytick.labelsize': gfontsize,
    'text.usetex': False,
    # 'figure.figsize': [10, 15]
    # 'ytick.major.pad': 4,
    # 'xtick.major.pad': 4,
    'font.family': 'Arial',
}
rcParams.update(params)

pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 1.0]
open_circle = mpl.path.Path(vert)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                  edgecolor='none', linewidth=0)

shapeList = []

v = np.r_[circ, circ[::-1] * 0.6]
oc = mpl.path.Path(v)

handles_a = [
    mlines.Line2D([0], [0], color='black', marker=oc,
                  markersize=6, label='Mean and SD'),
    mlines.Line2D([], [], linestyle='none', color='black', marker='*',
                  markersize=5, label='Median'),
    mlines.Line2D([], [], linestyle='none', markeredgewidth=1, marker='o',
                  color='black', markeredgecolor='w', markerfacecolor='black', alpha=0.5,
                  markersize=5, label='Single run')
]
handles_b = [
    mlines.Line2D([0], [1], color='black',  label='Mean'),
    Circle((0, 0), radius=1, facecolor='black', alpha=0.35, label='SD')
]


def pplots(data, ax, sub_colors=[], exp_title='', ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    sns.pointplot(data=np.transpose(data), palette=sub_colors,
                  size=5, estimator=np.mean,
                  ci='sd', capsize=0.2, linewidth=0.8, markers=[open_circle],
                  scale=1.6, ax=ax)

    sns.stripplot(data=np.transpose(data), edgecolor='white',
                  dodge=True, jitter=True,
                  alpha=.50, linewidth=0.8, size=5, palette=sub_colors, ax=ax)

    medians = []
    for d in data:
        medians.append([np.median(list(d))])
    sns.swarmplot(data=medians, palette=['#000000']*10,
                  marker='*', size=5,  ax=ax)


def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif


def angular_plot(data, experiments):
    num_experiments = len(data.keys())
    labels = []

    fig, ax = plt.subplots(num_experiments, 1, figsize=(
        8, 14), gridspec_kw={'width_ratios': [1]}, subplot_kw=dict(projection='polar'))
    fig.subplots_adjust(hspace=0.2, wspace=0.10)
    # sns.despine(bottom=True, left=True)

    ylim = [0, 0.14]
    for i, k in enumerate(sorted(data.keys())):
        vectors = data[k]
        labels.append(k)

        cax = ax
        if num_experiments > 1:
            cax = ax[i]

        cvector = []
        for v in vectors:
            phis = v[1:]
            phis_tm1 = v[:-1]
            phis = list(map(angle_to_pipi, phis-phis_tm1))
            cvector += phis

        bins_number = 180
        bins = np.linspace(-np.pi, np.pi, bins_number + 1)
        n, _, _ = cax.hist(cvector, bins, color=colors[i], alpha=0.95)
        # cax.set_xticks([np.pi/4, np.pi/4, 2*np.pi - np.pi/4])

    cax = ax
    if num_experiments > 1:
        cax = ax[0]   

    fig.text(0.5, 0.08, 'Angular velocity (m/s)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Frequency', ha='center',
             va='center', rotation='vertical')
    cax.legend(handles=shapeList, labels=labels,
               handletextpad=0.5, columnspacing=1,
               loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize)
    plt.savefig('angular_velocity.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resultant velocity histogram figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    args = parser.parse_args()

    experiments = {
        'Aggregated': '*_processed_velocities_filtered_twice.dat',
        # 'Hybrid': '*generated_velocities_filtered.dat',
        'Virtual': '*generated_virtu_velocities_filtered.dat',
        # 'Model': '*generated*velocities_filtered.dat',
        # 'Exp. 1': 'exp_1_processed_velocities_filtered.dat',
        # 'Exp. 2': 'exp_2_processed_velocities_filtered.dat',
        # 'Exp. 3': 'exp_3_processed_velocities_filtered.dat',
        # 'Exp. 4': 'exp_4_processed_velocities_filtered.dat',
        # 'Exp. 5': 'exp_5_processed_velocities_filtered.dat',
        # 'Exp. 6': 'exp_6_processed_velocities_filtered.dat',
        # 'Exp. 7': 'exp_7_processed_velocities_filtered.dat',
        # 'Exp. 8': 'exp_8_processed_velocities_filtered.dat',
        # 'Exp. 9': 'exp_9_processed_velocities_filtered.dat',
        # 'Exp. 10': 'exp_10_processed_velocities_filtered.dat',
        # 'real': '*processed_velocities_filtered.dat',
    }

    palette = sns.cubehelix_palette(len(experiments.keys()))
    colors = sns.color_palette(palette)

    for i in range(len(experiments.keys())):
        shapeList.append(Circle((0, 0), radius=1, facecolor=colors[i]))

    data = {}
    for e in sorted(experiments.keys()):
        data[e] = []
        vel = glob.glob(args.path + '/' + experiments[e])
        for v in vel:
            matrix = np.loadtxt(v)
            for i in range(matrix.shape[1] // 2):
                angles = np.arctan2(matrix[:, i*2+1], matrix[:, i*2]) 
                data[e].append(angles)

    angular_plot(data, experiments)
