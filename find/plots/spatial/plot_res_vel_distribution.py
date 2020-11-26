#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *

from find.utils.features import Velocities


from itertools import cycle

flatui = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

# palette = sns.cubehelix_palette(11)
# palette = sns.color_palette("Set1", n_colors=11, desat=.5)
# colors = sns.color_palette(palette)
colorcycler = cycle(flatui)

sns.set(style="darkgrid")

lines = ["-"]
linecycler = cycle(lines)
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


def linear_velocity_plot(data, experiments):
    num_experiments = len(data.keys())
    labels = []

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    for i, k in enumerate(sorted(data.keys())):
        vectors = data[k]
        labels.append(k)

        cvector = []
        for v in vectors:
            cvector += v.tolist()

        print(np.max(cvector))

        thres = []
        for v in cvector:
            if v < 0.5:
                thres.append(v)
        cvector = thres

        sns.kdeplot(cvector, ax=ax,
                    color=next(colorcycler), linestyle=next(linecycler), linewidth=1, label=k)

    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('KDE')

    ax.legend()
    # ax.legend(handles=shapeList, labels=labels,
    #           handletextpad=0.5, columnspacing=1,
    #           loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize)

    plt.savefig('linear_velocity.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resultant velocity histogram figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Timestep',
                        required=True)
    parser.add_argument('--radius', '-r', type=float,
                        help='Raidus',
                        default=0.25,
                        required=False)
    args = parser.parse_args()

    experiments = {
        'Hybrid': 'generated/*generated_positions.dat',
        'Virtual': 'generated/*generated_virtu_positions.dat',
        'Real': 'raw/*processed_positions.dat',
    }

    palette = sns.cubehelix_palette(len(experiments.keys()))
    colors = sns.color_palette(palette)

    for i in range(len(experiments.keys())):
        shapeList.append(Circle((0, 0), radius=1, facecolor=colors[i]))

    data = {}
    for e in sorted(experiments.keys()):
        pos = glob.glob(args.path + '/' + experiments[e])
        if len(pos) == 0:
            continue
        data[e] = []
        for p in pos:
            # TODO: this is to convert to meters but I should probably do this in a cleaner way
            matrix = np.loadtxt(p) * args.radius
            matrix = Velocities([matrix], args.timestep).get()[0]

            linear_velocity = np.array((matrix.shape[0], 1))
            for i in range(matrix.shape[1] // 2):
                linear_velocity = np.sqrt(matrix[:, i * 2] ** 2 + matrix[:, i * 2 + 1] ** 2
                                          - 2 * np.abs(matrix[:, i * 2]) * np.abs(matrix[:, i * 2 + 1]) * np.cos(
                    np.arctan2(matrix[:, i * 2 + 1], matrix[:, i * 2])))
                data[e].append(linear_velocity)

    linear_velocity_plot(data, experiments)
