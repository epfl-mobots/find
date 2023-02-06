#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
from find.utils.utils import compute_leadership
from find.plots.common import *

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def activity_plots(data, path, ax, args):
    percs = []
    for e in data.keys():
        samples = data[e]['samples']
        percs.append(samples[1] / samples[0])
    ax = sns.barplot(
        x=list(range(len(percs))), y=np.array(percs), palette='muted', ax=ax)
    return ax


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        data[e] = {}
        data[e]['samples'] = np.loadtxt(
            args.path + '/' + e + '/sample_counts.txt')

    _ = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    activity_plots(data, path, ax, args)

    ax.set_xticklabels(list(data.keys()))
    ax.legend()
    plt.savefig(path + 'activity_plot.png'.format(e))
