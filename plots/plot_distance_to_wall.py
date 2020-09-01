#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *
from itertools import cycle

from plot_geometrical_leader_info import compute_leadership
from utils.features import Velocities

flatui = ["#3498db", "#3498db", "#95a5a6", "#95a5a6",
          "#e74c3c", "#e74c3c", "#34495e", "#34495e", "#2ecc71", "#2ecc71"]
# palette = sns.cubehelix_palette(11)
# palette = sns.color_palette("Set1", n_colors=11, desat=.5)
# colors = sns.color_palette(palette)

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


def distance_plot(data, experiments):
    flatui = ["#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    colorcycler = cycle(flatui)

    lines = ["-"]
    linecycler = cycle(lines)

    num_experiments = len(data.keys())
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    labels = []
    for i, k in enumerate(sorted(data.keys())):
        labels.append(k)
        matrices = data[k]

        vectors = []
        for m in matrices:
            for j in range(m.shape[1]):
                vectors.append(list(m[:, j]))

        cvector = []
        for v in vectors:
            cvector += v

        sns.kdeplot(cvector, ax=ax,
                    color=next(colorcycler),
                    linestyle=next(linecycler), label=k, linewidth=1)

    ax.set_xlabel('Distance to wall (m)')
    ax.set_ylabel('KDE')
    ax.legend()
    # ax.legend(handles=shapeList, labels=labels,
    #           handletextpad=0.5, columnspacing=1,
    #           loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize)
    plt.savefig('distance.png', dpi=300)


def sep_distance_plot(distances, positions, args):
    flatui = ["#3498db", "#3498db", "#95a5a6", "#95a5a6",
              "#e74c3c", "#e74c3c", "#34495e", "#34495e", "#2ecc71", "#2ecc71"]
    lines = ["-", ":"]
    linecycler = cycle(lines)
    colorcycler = cycle(flatui)

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    num_experiments = len(distances.keys())

    leadership = {}
    for i, k in enumerate(sorted(data.keys())):
        pos = positions[k]
        vel = Velocities(pos, args.timestep).get()

        leadership[k] = []
        for idx in range(len(pos)):
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])

            # TODO: don't do it globally but locally and then reset the focal window
            # this is to sanitize the leadership timeseries
            window = 4
            hwindow = window // 2
            lt = np.array(leadership_timeseries)
            for l in range(0, lt.shape[0], hwindow):
                lb = max([0, l - hwindow])
                ub = min([l + hwindow, lt.shape[0]])

                snap = list(lt[lb:ub, 1])
                fel = max(set(snap), key=snap.count)
                lt[lb:ub, 1] = fel
                leadership_timeseries[lb:ub] = lt[lb:ub].tolist()

            leadership[k].append(leadership_timeseries)

    labels = []
    for i, k in enumerate(sorted(data.keys())):
        labels.append(k)
        distances = data[k]
        leaders = leadership[k]

        leader_dist = []
        follower_dist = []

        for idx in range(len(leaders)):
            leadership_mat = np.array(leaders[idx])
            dist_mat = distances[idx]

            num_individuals = dist_mat.shape[1]
            for j in range(num_individuals):
                idx_leaders = np.where(leadership_mat[:, 1] == j)

                leader_dist += dist_mat[idx_leaders, j].tolist()[0]
                follower_idcs = list(range(num_individuals))
                follower_idcs.remove(j)
                for fidx in follower_idcs:
                    follower_dist += dist_mat[idx_leaders, fidx].tolist()[0]

        sns.kdeplot(leader_dist, ax=ax, color=next(colorcycler),
                    linestyle=next(linecycler), label='Leader (' + k + ')', linewidth=1)
        sns.kdeplot(follower_dist, ax=ax, color=next(colorcycler),
                    linestyle=next(linecycler), label='Follower (' + k + ')', linewidth=1)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('KDE')
    ax.legend()
    # ax.legend(handles=shapeList, labels=labels,
    #            handletextpad=0.5, columnspacing=1,
    #            loc="upper right", ncol=1, framealpha=0, frameon=False, fontsize=gfontsize)
    plt.savefig('distance_leader_follower.png', dpi=300)


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
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    args = parser.parse_args()

    experiments = {
        'Hybrid': '*generated_positions.dat',
        'Virtual': '*generated_virtu_positions.dat',
        'Real': '*processed_positions.dat',
    }

    palette = sns.cubehelix_palette(len(experiments.keys()))
    colors = sns.color_palette(palette)

    for i in range(len(experiments.keys())):
        shapeList.append(Circle((0, 0), radius=1, facecolor=colors[i]))

    data = {}
    positions = {}
    for e in sorted(experiments.keys()):
        pos = glob.glob(args.path + '/' + experiments[e])
        if len(pos) == 0:
            continue
        data[e] = []
        positions[e] = []
        for v in pos:
            matrix = np.loadtxt(v) * args.radius
            distances = np.array((matrix.shape[0], 1))

            dist_mat = []
            for i in range(matrix.shape[1] // 2):
                distance = args.radius - \
                    np.sqrt(matrix[:, i * 2] ** 2 + matrix[:, i * 2 + 1] ** 2)
                dist_mat.append(distance)

            dist_mat = np.array(dist_mat).T
            data[e].append(dist_mat)
            positions[e].append(matrix)

    distance_plot(data, experiments)
    sep_distance_plot(data, positions, args)
