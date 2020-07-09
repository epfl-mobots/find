#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
palette = sns.cubehelix_palette(11)
colors = sns.color_palette(palette)
sns.set(style="darkgrid")

from itertools import cycle
lines = ["-","--","-.",":"]
linecycler = cycle(lines)

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

def distance_plot(data):
    num_experiments = len(data.keys())

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    labels = []
    for i, k in enumerate(sorted(data.keys())):
        labels.append(k)
        vectors = data[k]
        cvector = []
        for v in vectors:
            cvector += v.tolist()

        sns.kdeplot(cvector, ax=ax, color=colors[i], linestyle=next(linecycler))

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('KDE')

    ax.legend(handles=shapeList, labels=labels,
               handletextpad=0.5, columnspacing=1,
               loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize)
    plt.savefig('interindividual_distance.png', dpi=300)


def sep_distance_plot(data):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resultant velocity histogram figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--radius', '-r', type=float,
                        help='Raidus',
                        default=0.25,
                        required=False)
    parser.add_argument('--separate-leader', action='store_true',
                        help='Flag to plot leader and follower separately',
                        default=False)
    args = parser.parse_args()

    experiments = {
        'Real': '*_processed_positions.dat',
        'Hybrid': '*generated_positions.dat',
        'Virtual': '*generated_virtu_positions.dat',        
    }

    palette = sns.cubehelix_palette(len(experiments.keys()))
    colors = sns.color_palette(palette)

    for i in range(len(experiments.keys())):
        shapeList.append(Circle((0, 0), radius=1, facecolor=colors[i]))

    data = {}
    for e in sorted(experiments.keys()):
        data[e] = []
        pos = glob.glob(args.path + '/' + experiments[e])
        for v in pos:
            matrix = np.loadtxt(v) * args.radius
            distances = np.array((matrix.shape[0], 1))
            distance = np.sqrt((matrix[:, 0] - matrix[:, 2]) ** 2 + (matrix[:, 1] - matrix[:, 3]) ** 2)
            data[e].append(distance)

    distance_plot(data)
    if args.separate_leader:
        sep_distance_plot(data)
