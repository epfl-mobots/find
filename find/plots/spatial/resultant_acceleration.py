#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Accelerations, Velocities
from find.utils.utils import compute_leadership
from find.plots.common import *


def compute_resultant_acceleration(data, ax, args, clipping_range=[0.0, 1.75]):
    lines = ['--', ':']
    linecycler = cycle(lines)
    new_palette = uni_palette()
    new_palette *= 3
    ccycler = cycle(sns.color_palette(new_palette))

    if not args.robot:
        leadership = {}
        for k in sorted(data.keys()):
            p = data[k]['pos']
            v = data[k]['vel']
            leadership[k] = []
            for idx in range(len(p)):
                if p[idx].shape[1] // 2 > 1:
                    (_, leadership_timeseries) = compute_leadership(
                        p[idx], v[idx])
                    leadership[k].append(leadership_timeseries)

        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        labels = []
        for k in sorted(data.keys()):
            labels.append(k)
            leaders = leadership[k]
            acc = data[k]['acc']
            leader_dist = []
            follower_dist = []

            for idx in range(len(acc)):
                if len(leaders):
                    leadership_mat = np.array(leaders[idx])
                    num_individuals = acc[idx].shape[1]
                    for j in range(num_individuals):
                        idx_leaders = np.where(leadership_mat[:, 1] == j)
                        leader_dist += acc[idx][idx_leaders, j].tolist()[0]
                        follower_idcs = list(range(num_individuals))
                        follower_idcs.remove(j)
                        for fidx in follower_idcs:
                            follower_dist += acc[idx][idx_leaders,
                                                      fidx].tolist()[0]
                else:
                    leader_dist += acc[idx][:, 0].tolist()

            ls = next(linecycler)
            print('Accelerations', k)
            print('LF: ', np.mean(leader_dist+follower_dist),
                  np.std(leader_dist+follower_dist))
            print('L: ', np.mean(leader_dist),
                  np.std(leader_dist))
            print('F: ', np.mean(follower_dist),
                  np.std(follower_dist))
            ccolour = next(ccycler)

            # ax = sns.kdeplot(leader_dist + follower_dist, ax=ax, color=next(colorcycler),
            #                  linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0.0, 1.8], bw_adjust=0.15, cut=-1)
            ax = sns.kdeplot(leader_dist, ax=ax, color=ccolour,
                             linestyle='--', label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0.0, 1.8], bw_adjust=0.15, cut=-1)
            ax = sns.kdeplot(follower_dist, ax=ax, color=ccolour,
                             linestyle=':', label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0.0, 1.8], bw_adjust=0.15, cut=-1)
    else:
        for k in sorted(data.keys()):
            rvel = data[k]['acc']
            ridcs = data[k]['ridx']

            robot_dist = []
            neigh_dist = []
            separate_fish = False

            for idx in range(len(rvel)):
                ridx = ridcs[idx]
                num_individuals = rvel[idx].shape[1]
                if num_individuals == 2 and args.agents12 and ridx < 0:
                    ridx = 0
                    separate_fish = True

                for j in range(num_individuals):
                    if ridx >= 0 and ridx == j:
                        robot_dist += rvel[idx][:, ridx].tolist()
                    else:
                        neigh_dist += rvel[idx][:, j].tolist()

            ls = next(linecycler)
            print('Accelerations', k)
            if len(robot_dist):
                print('Robot: ', np.mean(robot_dist), np.std(robot_dist))
            print('Neighs: ', np.mean(neigh_dist), np.std(neigh_dist))

            ccolour = next(ccycler)

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
        if e == 'BOBI':
            timestep = args.bt
        elif e == 'F44':
            timestep = args.f44t
        else:
            timestep = args.timestep

        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        data[e]['acc'] = []
        if args.robot:
            data[e]['ridx'] = []

        for p in pos:
            positions = np.loadtxt(p) * args.radius
            velocities = Velocities([positions], timestep).get()[0]
            accelerations = Accelerations([velocities], timestep).get()[0]
            linear_acceleration = np.array((accelerations.shape[0], 1))
            tup = []
            for i in range(accelerations.shape[1] // 2):
                linear_acceleration = np.sqrt(accelerations[:, i * 2] ** 2 + accelerations[:, i * 2 + 1] ** 2
                                              - 2 * np.abs(accelerations[:, i * 2]) * np.abs(accelerations[:, i * 2 + 1]) * np.cos(
                    np.arctan2(accelerations[:, i * 2 + 1], accelerations[:, i * 2]))).tolist()
                tup.append(linear_acceleration)
            data[e]['acc'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    ax = compute_resultant_acceleration(data, ax, args)

    ax.set_xlabel(r'$\alpha$ ($m/s^2$)')
    ax.set_ylabel('PDF')
    ax.legend()
    ax.set_xlim([-0.09, 1.8])
    plt.savefig(path + 'linear_acceleration.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resultant acceleration histogram figure')
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
