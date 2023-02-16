#!/usr/bin/env python
import glob
import argparse

from find.utils.utils import angle_to_pipi, compute_leadership
from find.utils.features import Velocities
from find.plots.common import *

from copy import deepcopy


def relative_orientation_to_neigh(data, ax, args):
    lines = ['-']
    linecycler = cycle(uni_lines)
    new_palette = uni_palette()
    # for p in uni_palette():
    #     new_palette.extend([p, p, p])
    ccycler = cycle(sns.color_palette(new_palette))

    if not args.robot:
        labels = []
        leadership = {}

        leadership = {}
        for k in sorted(data.keys()):
            pos = data[k]['pos']
            vel = data[k]['vel']
            leadership[k] = []
            for idx in range(len(pos)):
                (_, leadership_timeseries) = compute_leadership(
                    pos[idx], vel[idx])
                leadership[k].append(leadership_timeseries)

        for k in sorted(data.keys()):
            leaders = leadership[k]
            labels.append(k)

            leader_dist = []
            follower_dist = []

            for e in range(len(data[k]['pos'])):
                p = data[k]['pos'][e]
                v = data[k]['vel'][e]

                hdgs = np.empty((p.shape[0], 0))
                for i in range(p.shape[1] // 2):
                    hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                    hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

                # for the focal
                angle_dif_focal = hdgs[:, 0] - hdgs[:, 1]
                angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
                angle_dif_focal = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_focal)))

                # for the neigh
                angle_dif_neigh = hdgs[:, 1] - hdgs[:, 0]
                angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
                angle_dif_neigh = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
                angle_difs = [angle_dif_focal, angle_dif_neigh]

                leadership_mat = np.array(leaders[e])
                for j in range(p.shape[1] // 2):
                    idx_leaders = np.where(leadership_mat[:, 1] == j)

                    leader_dist += angle_difs[j][idx_leaders].tolist()
                    follower_idcs = list(range(p.shape[1] // 2))
                    follower_idcs.remove(j)
                    for fidx in follower_idcs:
                        follower_dist += angle_difs[fidx][idx_leaders].tolist()

            print('Orientation to neigh', k)
            print('LF: ', np.mean(leader_dist+follower_dist),
                  np.std(leader_dist+follower_dist))

            ax = sns.kdeplot(leader_dist + follower_dist, ax=ax, color=next(ccycler),
                             linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.35, cut=-1)
            # sns.kdeplot(leader_dist, ax=ax, color=next(ccycler),
            #             linestyle=next(linecycler), label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.25, cut=-1)
            # sns.kdeplot(follower_dist, ax=ax, color=next(ccycler),
            #             linestyle=next(linecycler), label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.25, cut=-1)
    else:
        for k in sorted(data.keys()):
            ridcs = data[k]['ridx']
            robot_dist = []
            neigh_dist = []
            separate_fish = False

            for e in range(len(data[k]['pos'])):
                p = data[k]['pos'][e]
                v = data[k]['vel'][e]
                ridx = ridcs[e]

                hdgs = np.empty((p.shape[0], 0))
                for i in range(p.shape[1] // 2):
                    hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                    hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

                # for the focal
                angle_dif_focal = hdgs[:, 0] - hdgs[:, 1]
                angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
                angle_dif_focal = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_focal)))

                # for the neigh
                angle_dif_neigh = hdgs[:, 1] - hdgs[:, 0]
                angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
                angle_dif_neigh = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
                angle_difs = [np.abs(angle_dif_focal), np.abs(angle_dif_neigh)]

                num_individuals = p.shape[1] // 2
                if num_individuals == 2 and args.agents12 and ridx < 0:
                    ridx = 0
                    separate_fish = True

                for j in range(p.shape[1] // 2):
                    if ridx >= 0 and ridx == j:
                        robot_dist += angle_difs[ridx].tolist()
                    else:
                        neigh_dist += angle_difs[j].tolist()

            ls = next(linecycler)
            print('Orientation to wall', k)
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
                             linestyle=ls, label=label_neigh, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0, 180], bw_adjust=.35, cut=-1)
            if len(robot_dist):
                ax = sns.kdeplot(robot_dist, ax=ax, color=ccolour,
                                 linestyle=':', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0, 180], bw_adjust=.35, cut=-1)
                ax = sns.kdeplot(robot_dist+neigh_dist, ax=ax, color=ccolour,
                                 linestyle='-', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[0, 180], bw_adjust=.35, cut=-1)

    return ax


def relative_orientation_to_wall(data, ax, args):
    lines = ['--', ':']
    linecycler = cycle(lines)
    new_palette = uni_palette()
    new_palette *= 3
    ccycler = cycle(sns.color_palette(new_palette))

    if not args.robot:
        labels = []
        leadership = {}

        for k in sorted(data.keys()):
            num_individuals = data[k]['pos'][0].shape[1] // 2

            if num_individuals == 1:

                dist = []
                for e in range(len(data[k]['pos'])):
                    p = data[k]['pos'][e]
                    v = data[k]['vel'][e]

                    hdgs = np.empty((v.shape[0], 0))
                    for i in range(v.shape[1] // 2):
                        hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                        hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

                    # for the focal
                    angle_dif_focal = hdgs[:, 0] - np.arctan2(p[:, 1], p[:, 0])
                    angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
                    angle_dif_focal = np.array(list(
                        map(lambda x: x * 180 / np.pi, angle_dif_focal)))

                    dist += angle_dif_focal.tolist()

                ccolour = next(ccycler)
                ax = sns.kdeplot(dist, ax=ax, color=ccolour,
                                 linestyle='-', label='Single agent mean', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.15, cut=-1)
            else:
                pos = data[k]['pos']
                vel = data[k]['vel']
                leadership[k] = []
                for idx in range(len(pos)):
                    (_, leadership_timeseries) = compute_leadership(
                        pos[idx], vel[idx])
                    leadership[k].append(leadership_timeseries)
                leaders = leadership[k]
                labels.append(k)

                leader_dist = []
                follower_dist = []

                for e in range(len(data[k]['pos'])):
                    p = data[k]['pos'][e]
                    v = data[k]['vel'][e]

                    hdgs = np.empty((p.shape[0], 0))
                    for i in range(p.shape[1] // 2):
                        hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                        hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

                    # for the focal
                    angle_dif_focal = hdgs[:, 0] - np.arctan2(p[:, 1], p[:, 0])
                    angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
                    angle_dif_focal = np.array(list(
                        map(lambda x: x * 180 / np.pi, angle_dif_focal)))

                    # for the neigh
                    angle_dif_neigh = hdgs[:, 1] - np.arctan2(p[:, 3], p[:, 2])
                    angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
                    angle_dif_neigh = np.array(list(
                        map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
                    angle_difs = [angle_dif_focal, angle_dif_neigh]

                    leadership_mat = np.array(leaders[e])
                    for j in range(p.shape[1] // 2):
                        idx_leaders = np.where(leadership_mat[:, 1] == j)

                        leader_dist += angle_difs[j][idx_leaders].tolist()
                        follower_idcs = list(range(p.shape[1] // 2))
                        follower_idcs.remove(j)
                        for fidx in follower_idcs:
                            follower_dist += angle_difs[fidx][idx_leaders].tolist()

                print('Orientation to wall', k)
                print('LF: ', np.mean(leader_dist+follower_dist),
                      np.std(leader_dist+follower_dist))
                print('L: ', np.mean(leader_dist),
                      np.std(leader_dist))
                print('F: ', np.mean(follower_dist),
                      np.std(follower_dist))

                ccolour = next(ccycler)
                # ax = sns.kdeplot(leader_dist + follower_dist, ax=ax, color=next(ccycler),
                #                  linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.15, cut=-1)
                ax = sns.kdeplot(leader_dist, ax=ax, color=ccolour,
                                 linestyle='--', label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.15, cut=-1)
                ax = sns.kdeplot(follower_dist, ax=ax, color=ccolour,
                                 linestyle=':', label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.15, cut=-1)
    else:
        for k in sorted(data.keys()):
            ridcs = data[k]['ridx']
            robot_dist = []
            neigh_dist = []
            separate_fish = False

            for e in range(len(data[k]['pos'])):
                p = data[k]['pos'][e]
                v = data[k]['vel'][e]
                ridx = ridcs[e]

                hdgs = np.empty((p.shape[0], 0))
                for i in range(p.shape[1] // 2):
                    hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                    hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

                # for the focal
                angle_dif_focal = hdgs[:, 0] - np.arctan2(p[:, 1], p[:, 0])
                angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
                angle_dif_focal = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_focal)))

                # for the neigh
                angle_dif_neigh = hdgs[:, 1] - np.arctan2(p[:, 3], p[:, 2])
                angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
                angle_dif_neigh = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
                angle_difs = [np.abs(angle_dif_focal), np.abs(angle_dif_neigh)]

                num_individuals = p.shape[1] // 2
                if num_individuals == 2 and args.agents12 and ridx < 0:
                    ridx = 0
                    separate_fish = True

                for j in range(p.shape[1] // 2):
                    if ridx >= 0 and ridx == j:
                        robot_dist += angle_difs[ridx].tolist()
                    else:
                        neigh_dist += angle_difs[j].tolist()

            ls = next(linecycler)
            print('Orientation to wall', k)
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
                             linestyle=ls, label=label_neigh, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.35, cut=-1)
            if len(robot_dist):
                ax = sns.kdeplot(robot_dist, ax=ax, color=ccolour,
                                 linestyle=':', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.35, cut=-1)
                ax = sns.kdeplot(robot_dist+neigh_dist, ax=ax, color=ccolour,
                                 linestyle='-', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.35, cut=-1)

    return ax


def viewing_angle(data, ax, args):
    lines = ['--', ':']
    linecycler = cycle(lines)
    new_palette = uni_palette()
    ccycler = cycle(sns.color_palette(new_palette))

    if not args.robot:
        labels = []
        leadership = {}

        leadership = {}
        for k in sorted(data.keys()):
            pos = data[k]['pos']
            vel = data[k]['vel']
            leadership[k] = []
            for idx in range(len(pos)):
                (_, leadership_timeseries) = compute_leadership(
                    pos[idx], vel[idx])
                leadership[k].append(leadership_timeseries)

        for k in sorted(data.keys()):
            leaders = leadership[k]
            labels.append(k)

            leader_dist = []
            follower_dist = []

            for e in range(len(data[k]['pos'])):
                p = data[k]['pos'][e]
                v = data[k]['vel'][e]

                hdgs = np.empty((p.shape[0], 0))
                for i in range(p.shape[1] // 2):
                    hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                    hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

                # for the focal
                angle_dif_focal = hdgs[:, 0] - \
                    np.arctan2(p[:, 3] - p[:, 1], p[:, 2] - p[:, 0])
                angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
                angle_dif_focal = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_focal)))

                # for the neigh
                angle_dif_neigh = hdgs[:, 1] - \
                    np.arctan2(p[:, 1] - p[:, 3], p[:, 0] - p[:, 2])
                angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
                angle_dif_neigh = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
                angle_difs = [angle_dif_focal, angle_dif_neigh]

                leadership_mat = np.array(leaders[e])
                for j in range(p.shape[1] // 2):
                    idx_leaders = np.where(leadership_mat[:, 1] == j)

                    leader_dist += angle_difs[j][idx_leaders].tolist()
                    follower_idcs = list(range(p.shape[1] // 2))
                    follower_idcs.remove(j)
                    for fidx in follower_idcs:
                        follower_dist += angle_difs[fidx][idx_leaders].tolist()

            print('Viewing angle', k)
            print('LF: ', np.mean(leader_dist+follower_dist),
                  np.std(leader_dist+follower_dist))
            print('L: ', np.mean(leader_dist),
                  np.std(leader_dist))
            print('F: ', np.mean(follower_dist),
                  np.std(follower_dist))

            ccolour = next(ccycler)
            # ax = sns.kdeplot(leader_dist + follower_dist, ax=ax, color=ccolour,
            #                  linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-200, 200], bw_adjust=.25, cut=-1)
            ax = sns.kdeplot(leader_dist, ax=ax, color=ccolour,
                             linestyle='--', label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-200, 200], bw_adjust=.25, cut=-1)
            ax = sns.kdeplot(follower_dist, ax=ax, color=ccolour,
                             linestyle=':', label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-200, 200], bw_adjust=.25, cut=-1)
    else:
        for k in sorted(data.keys()):
            ridcs = data[k]['ridx']
            robot_dist = []
            neigh_dist = []
            separate_fish = False

            for e in range(len(data[k]['pos'])):
                p = data[k]['pos'][e]
                v = data[k]['vel'][e]
                ridx = ridcs[e]

                hdgs = np.empty((p.shape[0], 0))
                for i in range(p.shape[1] // 2):
                    hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                    hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

                # for the focal
                angle_dif_focal = hdgs[:, 0] - \
                    np.arctan2(p[:, 3] - p[:, 1], p[:, 2] - p[:, 0])
                angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
                angle_dif_focal = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_focal)))

                # for the neigh
                angle_dif_neigh = hdgs[:, 1] - \
                    np.arctan2(p[:, 1] - p[:, 3], p[:, 0] - p[:, 2])
                angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
                angle_dif_neigh = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
                angle_difs = [angle_dif_focal, angle_dif_neigh]

                num_individuals = p.shape[1] // 2
                if num_individuals == 2 and args.agents12 and ridx < 0:
                    ridx = 0
                    separate_fish = True

                for j in range(p.shape[1] // 2):
                    if ridx >= 0 and ridx == j:
                        robot_dist += angle_difs[ridx].tolist()
                    else:
                        neigh_dist += angle_difs[j].tolist()

            ls = next(linecycler)
            print('Viewing angle', k)
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
                             linestyle=ls, label=label_neigh, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.35, cut=-1)
            if len(robot_dist):
                ax = sns.kdeplot(robot_dist, ax=ax, color=ccolour,
                                 linestyle=':', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.35, cut=-1)
                ax = sns.kdeplot(robot_dist+neigh_dist, ax=ax, color=ccolour,
                                 linestyle='-', label=label_robot, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.35, cut=-1)

    return ax


def plot(exp_files, path, args):
    single_agent = False
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
        data[e] = {'pos': [], 'vel': []}
        for p in pos:
            p = np.loadtxt(p) * args.radius
            v = Velocities([p], timestep).get()[0]
            data[e]['pos'].append(p)
            data[e]['vel'].append(v)
            if p.shape[1] // 2 == 1:
                single_agent = True

    if not single_agent:
        # relative angle to neigh
        _ = plt.figure(figsize=(6, 5))
        ax = plt.gca()

        ax = relative_orientation_to_neigh(data, ax, args)

        ax.set_xlabel(r'$\Delta \phi$ (degrees)')
        ax.set_ylabel('PDF')
        ax.set_xticks(np.arange(-180, 181, 60))
        ax.legend()
        plt.savefig(path + 'phi.png')

    # relative angle to wall
    _ = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    ax = relative_orientation_to_wall(data, ax, args)

    ax.set_xlabel(r'$\theta_w$ (degrees)')
    ax.set_ylabel('PDF')
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.legend()
    plt.savefig(path + 'thetaw.png')

    if not single_agent:
        # viewing angle
        _ = plt.figure(figsize=(6, 5))
        ax = plt.gca()

        ax = viewing_angle(data, ax, args)

        ax.set_xlabel(r'$\psi$ (degrees)')
        ax.set_ylabel('PDF')
        ax.set_xticks(np.arange(-200, 201, 50))
        ax.legend()
        plt.savefig(path + 'psiw.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Relative orientation figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--radius', '-r', type=float,
                        help='Radius',
                        default=0.25,
                        required=False)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
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
