#!/usr/bin/env python
import os
import sys
import glob
import argparse
import numpy as np
from pathlib import Path
import seaborn as sns

import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.widgets import RangeSlider

fstop = 450
fstart = 0

# palette = ['#48A14D']
# palette = ['#1E81B0', '#7bc2e3']
# palette = ['#1E81B0', '#48A14D']
# palette = ['#48A14D', '#1E81B0']
# palette = ['#1E81B0', '#D51A3C']
palette = ['#D51A3C', '#1E81B0']


def plot(exp_files, path, args):
    data = {}

    fig = plt.figure(figsize=(5, 4))
    # fig.subplots_adjust(bottom=0.25)
    # fig, axs = plt.subplots(2, 1, figsize=(7, 5))
    ax = plt.gca()

    tfile = args.traj_visualisation_list[0]
    traj = np.loadtxt(tfile)
    samples = traj.shape[0]
    num_inds = traj.shape[1] // 2
    draw_point_fps = 30

    for i in range(num_inds):
        traj[:, i*2+1] = 1500 - traj[:, i*2+1]

    # Create the RangeSlider
    slider_ax = fig.add_axes([0.17, 0.01, 0.60, 0.03])
    slider = RangeSlider(slider_ax, "Frames", 0, samples // 30)

    # ax.plot(traj[:450, 0], traj[:450, 1], color=palette[0])
    # if num_inds == 2:
    #     ax.plot(traj[:450, 2], traj[:450, 3], color=palette[1])

    # ax.axis('off')
    ax.set_xlim([0, 1500])
    ax.set_ylim([0, 1500])
    plt.tight_layout()

    def update(val):
        global fstop, fstart
        val = list(map(int, val))
        val[0] *= 30
        val[1] *= 30
        fstart = val[0]
        fstop = val[1]
        ax.clear()
        ax.plot(traj[val[0]:val[1], 0],
                traj[val[0]:val[1], 1], color=palette[0])
        points = list(range(val[0], val[1]+1, draw_point_fps))
        ax.plot(traj[points[1:], 0],
                traj[points[1:], 1], linestyle='None', marker='o', markersize=3, linewidth=2, color=palette[0])
        ax.plot(traj[points[0], 0],
                traj[points[0], 1], linestyle='None', marker='x', markersize=3, linewidth=2, color=palette[0])

        if num_inds == 2:
            ax.plot(traj[val[0]:val[1], 2],
                    traj[val[0]:val[1], 3], color=palette[1])
            ax.plot(traj[points[1:], 2],
                    traj[points[1:], 3], linestyle='None', marker='o', markersize=3, linewidth=2, color=palette[1])
            ax.plot(traj[points[0], 2],
                    traj[points[0], 3], linestyle='None', marker='x', markersize=3, linewidth=2, color=palette[1])

        ax.set_xlim([0, 1500])
        ax.set_ylim([0, 1500])
        plt.tight_layout()

        fig.canvas.draw_idle()
    slider.on_changed(update)

    plt.show()
    plt.close('all')

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    ax.axis('off')
    ax.set_xlim([0, 1500])
    ax.set_ylim([0, 1500])
    plt.tight_layout()

    ax.plot(traj[fstart:fstop, 0], traj[fstart:fstop, 1])

    ax.plot(traj[fstart:fstop, 0],
            traj[fstart:fstop, 1], color=palette[0])
    points = list(range(fstart, fstop+1, draw_point_fps))
    ax.plot(traj[points[1:], 0],
            traj[points[1:], 1], linestyle='None', marker='o', markersize=3, linewidth=2, color=palette[0])
    ax.plot(traj[points[0], 0],
            traj[points[0], 1], linestyle='None', marker='x', markersize=3, linewidth=2, color=palette[0])

    if num_inds == 2:
        ax.plot(traj[fstart:fstop, 2],
                traj[fstart:fstop, 3], color=palette[1])
        ax.plot(traj[points[1:], 2],
                traj[points[1:], 3], linestyle='None', marker='o', markersize=3, linewidth=2, color=palette[1])
        ax.plot(traj[points[0], 2],
                traj[points[0], 3], linestyle='None', marker='x', markersize=3, linewidth=2, color=palette[1])

    plt.savefig(path + 'frame_{}-{}.png'.format(fstart, fstop),
                dpi=300, transparent=True)
