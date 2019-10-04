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
from random import randint
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


def distance_plot(data, experiments):
    num_experiments = len(data.keys())
    labels = []

    fig, ax = plt.subplots(num_experiments, 1, figsize=(
        8, 14), gridspec_kw={'width_ratios': [1]})
    fig.subplots_adjust(hspace=0.05, wspace=0.10)
    sns.despine(bottom=True, left=True)

    ylim = [0, 0.3]
    for i, k in enumerate(sorted(data.keys())):
        vectors = data[k]
        labels.append(k)

        cax = ax
        if num_experiments > 1:
            cax = ax[i]

        cvector = []
        for v in vectors:
            cvector += v.tolist()

        cax.hist(cvector, 50, [0.0, 0.5], weights=np.ones_like(
            cvector)/float(len(cvector)), color=colors[i])
        cax.set_ylim(ylim)
        if i != len(data.keys()) - 1:
            cax.set_xticklabels([])
        # cax.set_yticks(np.arange(0.02, 0.3, 0.02))
    cax = ax
    if num_experiments > 1:
        cax = ax[0]

    fig.text(0.5, 0.08, 'Distance (m)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Frequency', ha='center',
             va='center', rotation='vertical')
    cax.legend(handles=shapeList, labels=labels,
               handletextpad=0.5, columnspacing=1,
               loc="upper right", ncol=3, framealpha=0, frameon=False, fontsize=gfontsize)
    plt.savefig('distance_hist.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pairwise distance distribution')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    args = parser.parse_args()

    experiments = {
        'Original': '*_processed_positions_filtered.dat',
        'Virtual': '*generated_*_positions_filtered.dat',
    }

    palette = sns.cubehelix_palette(len(experiments.keys()) + 1)
    colors = sns.color_palette(palette)

    for i in range(len(experiments.keys()) + 1):
        shapeList.append(Circle((0, 0), radius=1, facecolor=colors[i]))

    positions = {}
    data = {}
    for e in sorted(experiments.keys()):
        data[e] = []
        positions[e] = []

        pos = glob.glob(args.path + '/' + experiments[e])
        for p in pos:
            matrix = np.loadtxt(p) * 0.29    
            dist = np.sqrt((matrix[:, 0] - matrix[:, 2]) ** 2 + (matrix[:, 1] - matrix[:, 3]) ** 2)
            data[e].append(dist)
            positions[e].append(matrix)

    # add randomized examples to see if the ML model is performing in a meaningful manner
    num_exps = len(data['Original'])
    data['Random'] = []
    for i in range(num_exps):
        idx1 = randint(0, num_exps-1)
        while True:
            idx2 = randint(0, num_exps-1)
            if idx1 != idx2:
                break

        p1 = positions['Original'][idx1]
        p2 = positions['Original'][idx2]
        data['Random'].append(np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2))
            
    distance_plot(data, experiments)
