#!/usr/bin/env python
import glob
import argparse

from find.utils.utils import angle_to_pipi, compute_leadership
from find.utils.features import Velocities
from find.plots.common import *


def distance_plot(data, positions, ax, args):
    lines = ['-', '--', ':']
    linecycler = cycle(lines)
    new_palette = []
    for p in uni_palette():
        new_palette.extend([p, p, p])
    colorcycler = cycle(sns.color_palette(new_palette))

    leadership = {}
    for k in sorted(data.keys()):
        pos = positions[k]
        vel = Velocities(pos, args.timestep).get()
        leadership[k] = []
        for idx in range(len(pos)):
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])
            leadership[k].append(leadership_timeseries)

    labels = []
    for k in sorted(data.keys()):
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

        ax = sns.kdeplot(leader_dist + follower_dist, ax=ax, color=next(colorcycler),
                         linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0.0, 0.6], bw_adjust=0.8, cut=-1)
        ax = sns.kdeplot(leader_dist, ax=ax, color=next(colorcycler),
                         linestyle=next(linecycler), label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0.0, 0.6], bw_adjust=0.8, cut=-1)
        ax = sns.kdeplot(follower_dist, ax=ax, color=next(colorcycler),
                         linestyle=next(linecycler), label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0.0, 0.6], bw_adjust=0.8, cut=-1)
    return ax


def plot(exp_files, path, args):
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

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    distance_plot(data, positions, ax, args)

    ax.set_xlabel(r'$r_i$ (m)')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig(path + 'distance_to_wall.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Distance to wall figure')
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
    parser.add_argument('--kde_gridsize',
                        type=int,
                        help='Grid size for kernel density estimation plots',
                        default=1500,
                        required=False)
    parser.add_argument('--type',
                        nargs='+',
                        default=['Real', 'Hybrid', 'Virtual'],
                        choices=['Real', 'Hybrid', 'Virtual'])
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
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
