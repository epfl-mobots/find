#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

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

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    for i, k in enumerate(sorted(data.keys())):
        vectors = data[k]
        labels.append(k)

        cvector = []
        for v in vectors:
            phis = v[1:]
            phis_tm1 = v[:-1]
            phis = list(map(angle_to_pipi, phis - phis_tm1))
            cvector += phis

        cvector = list(map(lambda x: x * 180 / np.pi, cvector))

        sns.kdeplot(cvector, ax=ax, color=colors[i], linestyle=next(linecycler))

    ax.set_xlabel('Angular change between successive timesteps (deg)')
    ax.set_ylabel('KDE')

    ax.legend(handles=shapeList, labels=labels,
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
        'Hybrid': '*generated_velocities.dat',
        'Virtual': '*generated_virtu_velocities.dat',
        'Real': '*processed_velocities.dat',
    }

    palette = sns.cubehelix_palette(len(experiments.keys()))
    colors = sns.color_palette(palette)

    for i in range(len(experiments.keys())):
        shapeList.append(Circle((0, 0), radius=1, facecolor=colors[i]))

    data = {}
    for e in sorted(experiments.keys()):
        vel = glob.glob(args.path + '/' + experiments[e])
        if len(vel) == 0:
            continue
        data[e] = []
        for v in vel:
            matrix = np.loadtxt(v)
            for i in range(matrix.shape[1] // 2):
                angles = np.arctan2(matrix[:, i * 2 + 1], matrix[:, i * 2])
                data[e].append(angles)

    angular_plot(data, experiments)
