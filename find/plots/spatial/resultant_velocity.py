#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
from find.utils.utils import compute_leadership
from find.plots.common import *

from scipy.stats import norm, rv_histogram


def compute_resultant_velocity(data, ax, args, clipping_range=[0.0, 0.4]):
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

        labels = []
        for k in sorted(data.keys()):
            labels.append(k)
            leaders = leadership[k]
            rvel = data[k]['rvel']
            leader_dist = []
            follower_dist = []

            for idx in range(len(rvel)):
                if len(leaders):
                    leadership_mat = np.array(leaders[idx])
                    num_individuals = rvel[idx].shape[1]
                    for j in range(num_individuals):
                        idx_leaders = np.where(leadership_mat[:, 1] == j)
                        leader_dist += rvel[idx][idx_leaders, j].tolist()[0]
                        follower_idcs = list(range(num_individuals))
                        follower_idcs.remove(j)
                        for fidx in follower_idcs:
                            follower_dist += rvel[idx][idx_leaders,
                                                       fidx].tolist()[0]
                else:
                    leader_dist += rvel[idx][:, 0].tolist()

            ls = next(linecycler)
            print('Velocities', k)
            print('LF: ', np.mean(leader_dist+follower_dist),
                  np.std(leader_dist+follower_dist))
            print('L: ', np.mean(leader_dist),
                  np.std(leader_dist))
            print('F: ', np.mean(follower_dist),
                  np.std(follower_dist))

            ccolour = next(ccycler)
            # ax = sns.kdeplot(leader_dist + follower_dist, ax=ax, color=ccolour,
            #                  linestyle'-', label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.5, cut=-1)
            ax = sns.kdeplot(leader_dist, ax=ax, color=ccolour,
                             linestyle='--', label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.6, cut=-1)
            ax = sns.kdeplot(follower_dist, ax=ax, color=ccolour,
                             linestyle=':', label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.6, cut=-1)
    else:
        for k in sorted(data.keys()):
            rvel = data[k]['rvel']
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
            print('Velocities', k)
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
                             linestyle=ls, label=label_neigh, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.6, cut=-1)
            if len(robot_dist):
                ax = sns.kdeplot(robot_dist, ax=ax, color=ccolour,
                                 linestyle=':', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.6, cut=-1)
                ax = sns.kdeplot(robot_dist+neigh_dist, ax=ax, color=ccolour,
                                 linestyle='-', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.6, cut=-1)

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
        data[e]['rvel'] = []
        for p in pos:
            if e == 'Virtual (Toulouse)':
                f = open(p)
                # to allow for loading fortran's doubles
                strarray = f.read().replace("D+", "E+").replace("D-", "E-")
                f.close()
                num_ind = len(strarray.split('\n')[0].strip().split('  '))
                positions = np.fromstring(
                    strarray, sep='\n').reshape(-1, num_ind) * args.radius
            elif e == 'Virtual (Toulouse cpp)':
                positions = np.loadtxt(p)[:, 2:] * args.radius
            else:
                positions = np.loadtxt(p) * args.radius
            velocities = Velocities([positions], timestep).get()[0]
            linear_velocity = np.array((velocities.shape[0], 1))
            tup = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)
            data[e]['rvel'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    ax = compute_resultant_velocity(data, ax, args, [0, 41])

    ax.set_xlabel('$V$')
    ax.set_ylabel('PDF')
    ax.set_xlim([-0.02, 0.4])
    ax.legend()
    plt.savefig(path + 'linear_velocity.png')


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
