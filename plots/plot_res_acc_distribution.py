#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *

from utils.features import Accelerations

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


def linear_acceleration_plot(data, experiments):
    num_experiments = len(data.keys())
    labels = []

    fig, ax = plt.subplots(num_experiments, 1, figsize=(
        8, 14), gridspec_kw={'width_ratios': [1]})
    fig.subplots_adjust(hspace=0.05, wspace=0.10)
    sns.despine(bottom=True, left=True)

    ylim = [0, 10]
    for i, k in enumerate(sorted(data.keys())):
        vectors = data[k]
        labels.append(k)

        cax = ax
        if num_experiments > 1:
            cax = ax[i]

        cvector = []
        for v in vectors:
            cvector += v.tolist()

        thres = []
        for v in cvector:
            if v < 4.0:
                thres.append(v)
        cvector = thres

        # sns.distplot(cvector, ax=cax, color=colors[i])
        # cax.hist(cvector, 125, [0.0, 2.5], weights=np.ones_like(
        #     cvector) / float(len(cvector)), color=colors[i])
        sns.distplot(cvector, ax=cax, color=colors[i], bins=225)
        cax.set_ylim(ylim)
        if i != len(data.keys()) - 1:
            cax.set_xticklabels([])
        # cax.set_yticks(np.arange(0.02, 0.21, 0.02))
    cax = ax
    if num_experiments > 1:
        cax = ax[0]

    # cax.set_xlabel('Velocity (m/s)')
    # cax.set_ylabel('Frequency')

    fig.text(0.5, 0.08, 'Acceleration (m/s^2)', ha='center', va='center')
    fig.text(0.06, 0.5, 'Frequency', ha='center',
             va='center', rotation='vertical')
    cax.legend(handles=shapeList, labels=labels,
               handletextpad=0.5, columnspacing=1,
               loc="upper right", ncol=3, framealpha=0, frameon=False, fontsize=gfontsize)
    plt.savefig('linear_acceleration.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resultant acceleration histogram figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Timestep',
                        required=True)
    args = parser.parse_args()

    experiments = {
        # 'Aggregated': '*_processed_velocities.dat',
        # 'Hybrid': '*generated_accelerations_filtered.dat',
        'Virtual': '*generated_velocities.dat',
        # 'Model': '*generated*accelerations_filtered.dat',
        # 'Exp. 1': 'exp_1_processed_accelerations_filtered.dat',
        # 'Exp. 2': 'exp_2_processed_accelerations_filtered.dat',
        # 'Exp. 3': 'exp_3_processed_accelerations_filtered.dat',
        # 'Exp. 4': 'exp_4_processed_accelerations_filtered.dat',
        # 'Exp. 5': 'exp_5_processed_accelerations_filtered.dat',
        # 'Exp. 6': 'exp_6_processed_accelerations_filtered.dat',
        # 'Exp. 7': 'exp_7_processed_accelerations_filtered.dat',
        # 'Exp. 8': 'exp_8_processed_accelerations_filtered.dat',
        # 'Exp. 9': 'exp_9_processed_accelerations_filtered.dat',
        # 'Exp. 10': 'exp_10_processed_accelerations_filtered.dat',
        'Real': '*processed_velocities.dat',
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
            # TODO: this is to convert to meters but I should probably do this in a cleaner way
            velocities = np.loadtxt(v) * 0.25
            acceleration = Accelerations(
                [velocities], args.timestep).get()[0]
            linear_acceleration = np.array((acceleration.shape[0], 1))

            for i in range(acceleration.shape[1] // 2):
                linear_acceleration = np.sqrt(acceleration[:, i * 2] ** 2 + acceleration[:, i * 2 + 1] ** 2
                                              - 2 * acceleration[:, i * 2] * acceleration[:, i * 2 + 1] * np.cos(
                    np.arctan2(acceleration[:, i * 2 + 1], acceleration[:, i * 2])))
                data[e].append(linear_acceleration)

    linear_acceleration_plot(data, experiments)
