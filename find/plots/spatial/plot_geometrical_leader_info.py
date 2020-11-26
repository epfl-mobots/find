#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *

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
    mlines.Line2D([0], [1], color='black', label='Mean'),
    Circle((0, 0), radius=1, facecolor='black', alpha=0.35, label='SD')
]


def bplots(data, ax, exp_title='', ticks=False):
    ax = sns.boxplot(data=data, width=0.25, notch=False,
                     whis=0.8, saturation=1, linewidth=1.0, ax=ax)
    # ax = sns.violinplot(data=data, width=0.25, notch=False, whis=1.5, saturation=1, ax=ax)
    # sns.swarmplot(data=data, edgecolor='white', linewidth=0.8, size=2)


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
    sns.swarmplot(data=medians, palette=['#000000'] * 10,
                  marker='*', size=5, ax=ax)


def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif


def compute_leadership(positions, velocities):
    ang0 = np.arctan2(positions[:, 1] - positions[:, 3],
                      positions[:, 0] - positions[:, 2])
    ang1 = np.arctan2(positions[:, 3] - positions[:, 1],
                      positions[:, 2] - positions[:, 0])
    theta = [ang1, ang0]

    previous_leader = -1
    leader_changes = -1
    leadership_timeseries = []

    for i in range(velocities.shape[0]):
        angles = []
        for j in range(velocities.shape[1] // 2):
            phi = np.arctan2(velocities[i, j * 2 + 1], velocities[i, j * 2])
            psi = angle_to_pipi(phi - theta[j][i])
            angles.append(np.abs(psi))

        geo_leader = np.argmax(angles)
        if geo_leader != previous_leader:
            leader_changes += 1
            previous_leader = geo_leader
        leadership_timeseries.append([i, geo_leader])

    return (leader_changes, leadership_timeseries)


def compute_consecutive(timeseries):
    occurences = []
    i = 1
    count = 1
    while i < timeseries.shape[0]:
        if timeseries[i] == timeseries[i-1]:
            count += 1
        else:
            occurences.append(count)
            count = 1
        i += 1
    occurences.append(count)
    return (np.mean(occurences), occurences)


def plot_consecutive(occurences, filename):
    fig = plt.figure(figsize=(15, 5))
    ax = plt.gca()

    bplots(occurences, ax)

    plt.savefig(
        filename + '.png',
        transparent=False,
        dpi=300
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--regex', action='store_true',
                        help='Flag to signify that args.positions is a regex',
                        default=False)
    args = parser.parse_args()

    files = []
    if not args.regex:
        files.append(args.path)
    else:
        files = glob.glob(args.path)

    leader_change_count = 0
    oc_list = []
    for f in files:
        vel = np.loadtxt(f)
        pos = np.loadtxt(f.replace('velocities', 'positions'))
        (leader_change_count, leadership_timeseries) = compute_leadership(pos, vel)
        leadership_timeseries = np.array(leadership_timeseries)
        np.savetxt(f.replace('velocities', 'leadership_info'),
                   leadership_timeseries)
        np.savetxt(f.replace('velocities', 'leadership_change_count'),
                   np.array([leader_change_count]))

        (avg, occurences) = compute_consecutive(leadership_timeseries[:, 1])
        oc_list.append(occurences)

    plot_consecutive(oc_list, 'geo_leader_bplot')
