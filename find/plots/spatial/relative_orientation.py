#!/usr/bin/env python
import glob
import argparse

from find.utils.utils import angle_to_pipi, compute_leadership
from find.utils.features import Velocities
from find.plots.common import *


def relative_orientation_to_neigh(data, experimentsm, path, args):
    # lines = ['-', '--', ':']
    linecycler = cycle(uni_lines)
    new_palette = uni_palette()
    # for p in uni_palette():
    #     new_palette.extend([p, p, p])
    ccycler = cycle(sns.color_palette(new_palette))

    _ = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    labels = []
    leadership = {}

    leadership = {}
    for k in sorted(data.keys()):
        pos = data[k]['pos']
        vel = data[k]['vel']
        leadership[k] = []
        for idx in range(len(pos)):
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])
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

        sns.kdeplot(leader_dist + follower_dist, ax=ax, color=next(ccycler),
                    linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.25, cut=-1)
        # sns.kdeplot(leader_dist, ax=ax, color=next(ccycler),
        #             linestyle=next(linecycler), label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.25, cut=-1)
        # sns.kdeplot(follower_dist, ax=ax, color=next(ccycler),
        #             linestyle=next(linecycler), label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.25, cut=-1)

    ax.set_xlabel(r'$\Delta \phi$ (degrees)')
    ax.set_ylabel('PDF')
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.legend()
    plt.savefig(path + 'relative_orientation.png')


def relative_orientation_to_wall(data, experimentsm, path, args):
    lines = ['-', '--', ':']
    linecycler = cycle(lines)
    new_palette = []
    for p in uni_palette():
        new_palette.extend([p, p, p])
    ccycler = cycle(sns.color_palette(new_palette))

    _ = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    labels = []
    leadership = {}

    leadership = {}
    for k in sorted(data.keys()):
        pos = data[k]['pos']
        vel = data[k]['vel']
        leadership[k] = []
        for idx in range(len(pos)):
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])
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

        sns.kdeplot(leader_dist + follower_dist, ax=ax, color=next(ccycler),
                    linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.15, cut=-1)
        sns.kdeplot(leader_dist, ax=ax, color=next(ccycler),
                    linestyle=next(linecycler), label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.15, cut=-1)
        sns.kdeplot(follower_dist, ax=ax, color=next(ccycler),
                    linestyle=next(linecycler), label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-180, 180], bw_adjust=.15, cut=-1)

    ax.set_xlabel(r'$\theta_w$ (degrees)')
    ax.set_ylabel('PDF')
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.legend()
    plt.savefig(path + 'relative_orientation_wall.png')


def viewing_angle(data, experimentsm, path, args):
    lines = ['-', '--', ':']
    linecycler = cycle(lines)
    new_palette = []
    for p in uni_palette():
        new_palette.extend([p, p, p])
    ccycler = cycle(sns.color_palette(new_palette))

    _ = plt.figure(figsize=(6, 5))
    ax = plt.gca()
    labels = []
    leadership = {}

    leadership = {}
    for k in sorted(data.keys()):
        pos = data[k]['pos']
        vel = data[k]['vel']
        leadership[k] = []
        for idx in range(len(pos)):
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])
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

        sns.kdeplot(leader_dist + follower_dist, ax=ax, color=next(ccycler),
                    linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-200, 200], bw_adjust=.25, cut=-1)
        sns.kdeplot(leader_dist, ax=ax, color=next(ccycler),
                    linestyle=next(linecycler), label='Leader (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-200, 200], bw_adjust=.25, cut=-1)
        sns.kdeplot(follower_dist, ax=ax, color=next(ccycler),
                    linestyle=next(linecycler), label='Follower (' + k + ')', linewidth=uni_linewidth, gridsize=args.kde_gridsize, clip=[-200, 200], bw_adjust=.25, cut=-1)

    ax.set_xlabel(r'$\psi$ (degrees)')
    ax.set_ylabel('PDF')
    ax.set_xticks(np.arange(-200, 201, 50))
    ax.legend()
    plt.savefig(path + 'viewing_angle.png')


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {'pos': [], 'vel': []}
        for p in pos:
            p = np.loadtxt(p) * args.radius
            v = Velocities([p], args.timestep).get()[0]
            data[e]['pos'].append(p)
            data[e]['vel'].append(v)
    relative_orientation_to_neigh(data, exp_files, path, args)
    relative_orientation_to_wall(data, exp_files, path, args)
    viewing_angle(data, exp_files, path, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Relative orientation figure')
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
