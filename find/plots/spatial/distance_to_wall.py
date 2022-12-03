#!/usr/bin/env python
import glob
import argparse

from find.utils.utils import angle_to_pipi, compute_leadership
from find.utils.features import Velocities
from find.plots.common import *


def distance_plot(data, ax, args, clipping_range=[0.0, 0.25]):
    lines = ['--', ':']
    linecycler = cycle(lines)
    new_palette = uni_palette()
    new_palette *= 3
    colorcycler = cycle(sns.color_palette(new_palette))

    if not args.robot:

        labels = []
        leadership = {}
        for k in sorted(data.keys()):
            labels.append(k)
            distances = data[k]['distance_to_wall']
            num_individuals = distances[0].shape[1]

            if num_individuals == 1:
                dist = []
                for i in range(len(distances)):
                    dist += distances[i][:, 0].tolist()

                ccolour = next(colorcycler)
                ax = sns.kdeplot(dist, ax=ax, color=ccolour,
                                 linestyle='-', label='Single agent mean', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=1.5)

            else:
                # leadership computations
                pos = data[k]['pos']
                vel = Velocities(pos, args.timestep).get()
                leadership[k] = []
                for idx in range(len(pos)):
                    (_, leadership_timeseries) = compute_leadership(
                        pos[idx], vel[idx])
                    leadership[k].append(leadership_timeseries)

                # split into folower/leader
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
                            follower_dist += dist_mat[idx_leaders,
                                                      fidx].tolist()[0]

                print('Dist to wall', k)
                print('LF: ', np.mean(leader_dist+follower_dist),
                      np.std(leader_dist+follower_dist))
                print('L: ', np.mean(leader_dist),
                      np.std(leader_dist))
                print('F: ', np.mean(follower_dist),
                      np.std(follower_dist))

                ccolour = next(colorcycler)
                # ax = sns.kdeplot(leader_dist + follower_dist, ax=ax, color=ccolour,
                #  linestyle = next(linecycler), label = k, linewidth = uni_linewidth, gridsize = args.kde_gridsize, clip = clipping_range, bw_adjust = 0.8, cut = -1)
                ax = sns.kdeplot(leader_dist, ax=ax, color=ccolour,
                                 linestyle='--', label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.8, cut=-1)
                ax = sns.kdeplot(follower_dist, ax=ax, color=ccolour,
                                 linestyle=':', label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.8, cut=-1)
    else:
        for k in sorted(data.keys()):
            distances = data[k]['distance_to_wall']
            ridcs = data[k]['ridx']

            robot_dist = []
            neigh_dist = []
            separate_fish = False

            for idx in range(len(distances)):
                dist_mat = distances[idx]
                ridx = ridcs[idx]
                num_individuals = dist_mat.shape[1]

                if num_individuals == 2 and args.agents12 and ridx < 0:
                    ridx = 0
                    separate_fish = True

                for j in range(num_individuals):
                    if ridx >= 0 and ridx == j:
                        robot_dist += dist_mat[:, ridx].tolist()
                    else:
                        neigh_dist += dist_mat[:, j].tolist()

            ls = next(linecycler)
            print('Distance to wall', k)
            if len(robot_dist):
                print('Robot: ', np.mean(robot_dist), np.std(robot_dist))
            print('Neighs: ', np.mean(neigh_dist), np.std(neigh_dist))

            ccolour = next(colorcycler)

            neigh_num = ''
            if separate_fish:
                neigh_num = ' 1'
            label_neigh = 'Fish{} ({})'.format(neigh_num, k)

            if separate_fish:
                label_robot = 'Fish 2 ({})'.format(neigh_num, k)
            else:
                label_robot = 'Robot ({})'.format(neigh_num, k)

            if len(robot_dist) == 0 and not separate_fish:
                ls = '-'
            else:
                ls = '--'

            ax = sns.kdeplot(neigh_dist, ax=ax, color=ccolour,
                             linestyle=ls, label=label_neigh, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.8, cut=-1)
            if len(robot_dist):
                ax = sns.kdeplot(robot_dist, ax=ax, color=ccolour,
                                 linestyle=':', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.8, cut=-1)
                ax = sns.kdeplot(robot_dist+neigh_dist, ax=ax, color=ccolour,
                                 linestyle='-', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.8, cut=-1)
    return ax


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['distance_to_wall'] = []
        data[e]['pos'] = []
        for p in pos:
            matrix = np.loadtxt(p) * args.radius
            dist_mat = []

            for i in range(matrix.shape[1] // 2):
                distance = args.radius - \
                    np.sqrt(matrix[:, i * 2] ** 2 + matrix[:, i * 2 + 1] ** 2)
                dist_mat.append(distance)
            dist_mat = np.array(dist_mat).T
            data[e]['distance_to_wall'].append(dist_mat)
            data[e]['pos'].append(matrix)

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    distance_plot(data, ax, args)

    ax.set_xlabel(r'$r_w$')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig(path + 'rw.png')


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
    for t in args.type:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
