#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
from find.utils.utils import compute_leadership
from find.plots.common import *

import numpy as np

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def activity_plots(data, path, ax, args, palette=['#1e81b0', '#D61A3C', '#48A14D'], freezing=False):
    percs = []
    for e in data.keys():
        samples = data[e]['samples']
        if freezing:
            percs.append((1-samples[1] / samples[0])*100)
        else:
            percs.append((samples[1] / samples[0]) * 100)

    npalette = palette

    if len(percs) == 5:
        if 'Biomimetic' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[2], palette[0], palette[0], palette[0], palette[0]]
        else:
            npalette = [palette[0]] * 5

    else:
        if 'Arbitrary' in data.keys() and 'Biomimetic' in data.keys() and 'Fish' not in data.keys():
            npalette = [palette[1], palette[2]]

        if 'Arbitrary' not in data.keys() and 'Biomimetic' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[2], palette[0]]

        if 'Biomimetic' not in data.keys() and 'Arbitrary' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[1], palette[0]]

        if 'Arbitrary' in data.keys() and 'Biomimetic' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[1], palette[2], palette[0]]

    ax = sns.barplot(
        x=list(range(len(percs))), y=np.array(percs), palette=npalette, ax=ax)

    # ax = ax.bar(np.arange(len(percs)), percs)

    patches = ax.patches
    for i in range(len(patches)):
        x = patches[i].get_x() + patches[i].get_width()/2
        y = patches[i].get_height() + 3
        ax.annotate('{:.1f}%'.format(percs[i]), (x, y), ha='center')

    return ax


def plot(exp_files, path, args, palette=['#1e81b0', '#D61A3C', '#48A14D']):
    data = {}
    for e in sorted(exp_files.keys()):
        data[e] = {}
        data[e]['samples'] = np.loadtxt(
            args.path + '/' + e + '/sample_counts.txt')

    _ = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    activity_plots(data, path, ax, args, palette=palette)

    ax.set_xticklabels(list(data.keys()))
    ax.legend()
    plt.savefig(path + 'activity_plot.png'.format(e))
