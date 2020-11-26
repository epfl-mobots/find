#!/usr/bin/env python
import glob
import argparse

from find.utils.utils import angle_to_pipi
from find.utils.features import Velocities
from find.plots.plot_common_utils import *


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


def distance_plot(data, experiments, path):
    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    labels = []
    for k in sorted(data.keys()):
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
                    color=next(uni_cycler),
                    linestyle='-', label=k, linewidth=1)

    ax.set_xlabel('Distance to wall (m)')
    ax.set_ylabel('KDE')
    ax.legend()
    plt.savefig(path + 'distance.png')


def sep_distance_plot(data, positions, path, args):
    lines = ["-", ":"]
    linecycler = cycle(lines)

    new_palette = []
    for p in uni_palette:
        new_palette.extend([p, p])
    colorcycler = cycle(sns.color_palette(new_palette))

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    leadership = {}
    for k in sorted(data.keys()):
        pos = positions[k]
        vel = Velocities(pos, args.timestep).get()
        leadership[k] = []
        for idx in range(len(pos)):
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])
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
    plt.savefig(path + 'distance_leader_follower.png', dpi=300)


def plot(exp_files, colours, path, args):
    data = {}
    positions = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = []
        positions[e] = []
        for p in pos:
            matrix = np.loadtxt(p) * args.radius
            dist_mat = []
            for i in range(matrix.shape[1] // 2):
                distance = args.radius - \
                    np.sqrt(matrix[:, i * 2] ** 2 + matrix[:, i * 2 + 1] ** 2)
                dist_mat.append(distance)
            dist_mat = np.array(dist_mat).T
            data[e].append(dist_mat)
            positions[e].append(matrix)

    # distance_plot(data, exp_files, path)
    if (positions[list(data.keys())[0]][0].shape[1] // 2) > 1:
        sep_distance_plot(data, positions, path, args)


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
    parser.add_argument('--type',
                        nargs='+',
                        default=['Original', 'Hybrid', 'Virtual'],
                        choices=['Original', 'Hybrid', 'Virtual'])
    parser.add_argument('--original_files',
                        type=str,
                        default='raw/*processed_positions.dat',
                        required=False)
    parser.add_argument('--hybrid_files',
                        type=str,
                        default='generated/*generated_positions.dat',
                        required=False)
    parser.add_argument('--virtual_files',
                        type=str,
                        default='generated/*generated_virtu_positions.dat',
                        required=False)
    args = parser.parse_args()

    exp_files = {}
    for t in args.types:
        if t == 'Original':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, uni_colours, './', args)
