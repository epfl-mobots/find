#!/usr/bin/env python
import os
import glob
import argparse

from numpy.lib.function_base import angle
from find.plots.spatial.relative_orientation import viewing_angle

from find.utils.utils import angle_to_pipi
from find.utils.features import Velocities
from find.plots.common import *


def gen_arrow_head_marker(angle):
    arr = np.array([[.1, .3], [.1, -.3], [1, 0]])  # arrow shape
    rot_mat = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))

    arrow_head_marker = mpl.path.Path(arr)
    return arrow_head_marker, scale


def create_dirs(fullpath):
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)


def compute_grid(data, args):
    grid = {}
    for k in sorted(data.keys()):
        grid[k] = {}
        skipped_count = 0
        total_count = 0

        trajectory_segments = []
        segment_len = args.observation_len + \
            args.prediction_len + 1  # +1 for the starting point

        # generate grids
        r_grid = np.arange(0, args.radius + 0.001, args.radius_grid_res)
        idx_dict = {}
        for i in range(1, len(r_grid)):
            idx_dict[(r_grid[i-1], r_grid[i])] = {}
            idx_dict[(r_grid[i-1], r_grid[i])]['all'] = []

        a_grid = np.arange(0, 180 + 0.001, args.angle_grid_res)
        idist_grid = np.arange(0, args.radius + 0.001, args.interdist_grid_res)
        psi_grid = np.arange(0, 180 + 0.001, args.viewing_angle_grid_res)

        for e in range(len(data[k]['pos'])):
            p = data[k]['pos'][e]
            v = data[k]['vel'][e]
            idist = data[k]['dist'][e]
            dwall = data[k]['dist_to_wall'][e]

            total_count += 1
            if segment_len > p.shape[0]:
                skipped_count += 1
                continue  # not enough samples to generate the plot

            # headings
            hdgs = np.empty((p.shape[0], 0))
            for i in range(p.shape[1] // 2):
                hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

            # angles to the wall
            angles_to_wall = np.empty((hdgs.shape[0], hdgs.shape[1]))
            for i in range(angles_to_wall.shape[1]):
                aw = hdgs[:, i] - \
                    np.arctan2(p[:, i*2+1], p[:, i*2])
                angles_to_wall[:, i] = list(map(angle_to_pipi, aw))

            # viewing angles
            angle_dif_focal = hdgs[:, 0] - \
                np.arctan2(p[:, 3] - p[:, 1], p[:, 2] - p[:, 0])
            angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))

            angle_dif_neigh = hdgs[:, 1] - \
                np.arctan2(p[:, 1] - p[:, 3], p[:, 0] - p[:, 2])
            angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))

            viewing_angles = np.array([angle_dif_focal, angle_dif_neigh]).T

            # complete matrix with info about the trajectories of the neighbouring fish
            matrix = np.hstack((p, v, angles_to_wall, dwall,
                               idist.reshape(-1, 1), viewing_angles, hdgs))
            for i in range(matrix.shape[0]):
                if i + segment_len >= matrix.shape[0]:
                    break

                # X1 Y1 X2 Y2 VX1 VY1 VX2 VY2 Th1 Th2 D1 D2 Idist Psi1 Psi2 Hdg1 Hdg2
                # 0  1  2  3  4   5   6   7   8   9   10 11 12    13   14   15  16
                trajectory_segments.append(matrix[i:(i+segment_len), :])

        # --- generate grids

        # radii grid
        for ts_idx, ts in enumerate(trajectory_segments):
            for i in range(1, len(r_grid)):
                if r_grid[i-1] <= ts[args.observation_len, 10] and ts[args.observation_len, 10] < r_grid[i]:
                    idx_dict[(r_grid[i-1], r_grid[i])
                             ]['all'].append((ts_idx, 0))
                if r_grid[i-1] <= ts[args.observation_len, 11] and ts[args.observation_len, 11] < r_grid[i]:
                    idx_dict[(r_grid[i-1], r_grid[i])
                             ]['all'].append((ts_idx, 1))

        # radii and angle to the wall
        for dgrid, sub_dict in idx_dict.items():
            for idx in sub_dict['all']:
                for i in range(1, len(a_grid)):
                    ts = trajectory_segments[idx[0]]

                    if (a_grid[i-1], a_grid[i]) not in sub_dict.keys():
                        idx_dict[dgrid][(a_grid[i-1], a_grid[i])] = {}
                        # ! this could be moved to not create empty lists if there are no idcs in it
                        idx_dict[dgrid][(a_grid[i-1], a_grid[i])]['all'] = []

                    if a_grid[i-1] <= np.abs(ts[args.observation_len, 8 + idx[1]] * 180 / np.pi) and np.abs(ts[args.observation_len, 8 + idx[1]] * 180 / np.pi) < a_grid[i]:
                        idx_dict[dgrid][(a_grid[i-1], a_grid[i])
                                        ]['all'].append(idx)

        # interindividual distance grid
        for dgrid, sub_dict1 in idx_dict.items():
            if dgrid == 'all':
                continue

            for agrid, sub_dict2 in sub_dict1.items():
                if agrid == 'all':
                    continue

                for idx in sub_dict2['all']:
                    for i in range(1, len(idist_grid)):
                        ts = trajectory_segments[idx[0]]

                        if (idist_grid[i-1], idist_grid[i]) not in sub_dict2.keys():
                            idx_dict[dgrid][agrid][(
                                idist_grid[i-1], idist_grid[i])] = {}
                            # ! this could be moved to not create empty lists if there are no idcs in it
                            idx_dict[dgrid][agrid][(
                                idist_grid[i-1], idist_grid[i])]['all'] = []

                        if idist_grid[i-1] <= ts[args.observation_len, 12] and ts[args.observation_len, 12] < idist_grid[i]:
                            idx_dict[dgrid][agrid][(
                                idist_grid[i-1], idist_grid[i])]['all'].append(idx)

        # viewing angle grid
        for dgrid, sub_dict1 in idx_dict.items():
            if dgrid == 'all':
                continue

            for agrid, sub_dict2 in sub_dict1.items():
                if agrid == 'all':
                    continue

                for idist_grid, sub_dict3 in sub_dict2.items():
                    if idist_grid == 'all':
                        continue

                    for idx in sub_dict3['all']:
                        for i in range(1, len(psi_grid)):
                            ts = trajectory_segments[idx[0]]

                            focal = idx[1]
                            if focal == 0:
                                neigh = 1
                                va_idx = 14  # we want the viewing angle of the neighbour to the focal
                            else:
                                neigh = 0
                                va_idx = 13

                            if (psi_grid[i-1], psi_grid[i]) not in sub_dict3.keys():
                                idx_dict[dgrid][agrid][idist_grid][(
                                    psi_grid[i-1], psi_grid[i])] = {}
                                # ! this could be moved to not create empty lists if there are no idcs in it
                                idx_dict[dgrid][agrid][idist_grid][(
                                    psi_grid[i-1], psi_grid[i])]['all'] = []

                            if psi_grid[i-1] <= ts[args.observation_len, va_idx] * 180 / np.pi and ts[args.observation_len, va_idx] * 180 / np.pi < psi_grid[i]:
                                idx_dict[dgrid][agrid][idist_grid][(
                                    psi_grid[i-1], psi_grid[i])]['all'].append(idx)

        grid[k]['idx_grid'] = idx_dict
        grid[k]['seg'] = trajectory_segments
        print('{} skipped: {} / {} experiment files'.format(k,
              skipped_count, total_count))
    return grid


def future_trajectory_variance(data, path, ax, args):
    grid = compute_grid(data, args)

    lines = ['-', '--', ':']
    linecycler = cycle(lines)
    new_palette = uni_palette()
    ccycler = cycle(sns.color_palette(new_palette))

    for k in sorted(grid.keys()):
        if k == 'Hybrid':
            lines = [':']
            linecycler = cycle(lines)
        elif k == 'Virtual':
            lines = ['--']
            linecycler = cycle(lines)
        elif k == 'Real':
            lines = ['-']
            linecycler = cycle(lines)

        idx_dict = grid[k]['idx_grid']
        segments = grid[k]['seg']

        # # plot distance barplots
        # _ = plt.figure(figsize=(6, 6))
        # ax = plt.gca()

        # cell_range = []
        # count = []
        # for dgrid, sub_dict1 in idx_dict.items():
        #     if dgrid == 'all':
        #         continue

        #     lb = '{:.3f}'.format(round(dgrid[0], 3))
        #     ub = '{:.3f}'.format(round(dgrid[1], 3))
        #     cell_range.append('[{}, {}]'.format(lb, ub))
        #     count.append(len(sub_dict1['all']))

        # sns.barplot(x=cell_range, y=count, ax=ax, color=next(ccycler))
        # ax.set_xlabel('Radius range (m)')
        # ax.set_ylabel('Number of trajectories')
        # ax.set_title(k)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig(
        #     path + 'dist_step__{}__type_{}.png'.format(args.radius_grid_res, k))

        # # plot angles to the wall barplots
        # for dgrid, sub_dict1 in idx_dict.items():
        #     if dgrid == 'all':
        #         continue

        #     cell_range = []
        #     angles_rad = []
        #     count = []

        #     for agrid, sub_dict2 in sub_dict1.items():
        #         if agrid == 'all':
        #             continue

        #         cell_range.append('[{}, {}]'.format(agrid[0], agrid[1]))
        #         angles_rad.append(agrid[1] * np.pi / 180.)
        #         count.append(len(sub_dict2['all']))

        #     _ = plt.figure(figsize=(6, 6))
        #     ax = plt.gca()
        #     ccycler = cycle(sns.color_palette(new_palette))

        #     ax = plt.subplot(projection='polar')
        #     ax.set_thetamin(0)
        #     ax.set_thetamax(180)

        #     if len(count):
        #         ax.bar(angles_rad, count, width=args.angle_grid_res *
        #                np.pi / 180., bottom=0.0, color=next(ccycler), alpha=1.0)
        #     ax.set_xlabel('Radius range (m)')
        #     ax.set_ylabel('Number of trajectories')
        #     ax.set_title(k)
        #     plt.tight_layout()

        #     lb = '{:.3f}'.format(round(dgrid[0], 3))
        #     ub = '{:.3f}'.format(round(dgrid[1], 3))
        #     dgrid_str = '{}-{}'.format(
        #         lb.replace('.', '_'), ub.replace('.', '_'))

        #     new_path = path + dgrid_str + '/'
        #     create_dirs(new_path)
        #     plt.savefig(
        #         new_path + 'theta_{}__type_{}.png'.format(dgrid_str, k))
        #     plt.close()

        # # inter individual distance plots
        # for dgrid, sub_dict1 in idx_dict.items():
        #     if dgrid == 'all':
        #         continue

        #     for agrid, sub_dict2 in sub_dict1.items():
        #         if agrid == 'all':
        #             continue

        #         cell_range = []
        #         count = []

        #         for igrid, sub_dict3 in sub_dict2.items():
        #             if igrid == 'all':
        #                 continue

        #             cell_range.append('[{:.3f}, {:.3f}]'.format(
        #                 round(igrid[0], 3), round(igrid[1], 3)))
        #             count.append(len(sub_dict3['all']))

        #         _ = plt.figure(figsize=(6, 6))
        #         ax = plt.gca()
        #         ccycler = cycle(sns.color_palette(new_palette))

        #         if len(count):
        #             sns.barplot(x=cell_range, y=count,
        #                         ax=ax, color=next(ccycler))
        #         ax.set_xlabel('Interindividual distance (m)')
        #         ax.set_ylabel('Number of trajectories')
        #         ax.set_title(k)
        #         plt.xticks(rotation=45)
        #         plt.tight_layout()

        #         lb = '{:.3f}'.format(round(dgrid[0], 3))
        #         ub = '{:.3f}'.format(round(dgrid[1], 3))
        #         dgrid_str = '{}-{}'.format(
        #             lb.replace('.', '_'), ub.replace('.', '_'))

        #         lb = '{:.3f}'.format(round(agrid[0], 3))
        #         ub = '{:.3f}'.format(round(agrid[1], 3))
        #         agrid_str = '{}-{}'.format(
        #             lb.replace('.', '_'), ub.replace('.', '_'))

        #         new_path = path + dgrid_str + '/' + agrid_str + '/'
        #         create_dirs(new_path)
        #         plt.savefig(
        #             new_path + 'idist_{}__{}__type_{}.png'.format(dgrid_str, agrid_str, k))
        #         plt.close()

        # trajectory variance plots
        for dgrid, sub_dict1 in idx_dict.items():
            if dgrid == 'all':
                continue

            for agrid, sub_dict2 in sub_dict1.items():
                if agrid == 'all':
                    continue

                cell_range = []
                count = []

                for igrid, sub_dict3 in sub_dict2.items():
                    if igrid == 'all':
                        continue

                    cell_range.append('[{:.3f}, {:.3f}]'.format(
                        round(igrid[0], 3), round(igrid[1], 3)))
                    count.append(len(sub_dict3['all']))

                    ccycler = cycle(sns.color_palette(new_palette))

                    for t in range(0, len(sub_dict3['all']), 1):

                        _ = plt.figure(figsize=(6, 6))
                        ax = plt.gca()

                        for idx in sub_dict3['all'][t:(t+1)]:
                            print('{}-{}-{}'.format(dgrid, agrid, igrid))
                            seg = segments[idx[0]]
                            r1 = np.sqrt((seg[:, idx[1]*2] - args.center[0]) ** 2 +
                                         (seg[:, idx[1]*2 + 1] - args.center[1]) ** 2)
                            phi1 = np.arctan2(
                                (seg[:, idx[1]*2 + 1] - args.center[1]), (seg[:, idx[1]*2] - args.center[0]))

                            # symmetrize
                            from copy import deepcopy
                            # - r1[args.observation_len]
                            new_r1 = deepcopy(r1)
                            new_phi1 = deepcopy(phi1) - phi1[0]

                            # new_r1[args.observation_len] = 0
                            if seg[args.observation_len, 15 + idx[1]] < 0:
                                print('in')
                                new_phi1 = -new_phi1
                                # phi1 = np.array(list(map(angle_to_pipi, phi1)))

                            new_phi1 = np.array(
                                list(map(angle_to_pipi, new_phi1)))

                            x = new_r1 * np.cos(new_phi1)
                            y = new_r1 * np.sin(new_phi1)

                            xo = r1 * np.cos(phi1)
                            yo = r1 * np.sin(phi1)

                            m1, scale = gen_arrow_head_marker(
                                angle_to_pipi(seg[0, 15 + idx[1]] - phi1[0]))
                            markersize = 5

                            c = next(ccycler)
                            print('n {}'.format(seg[0, 15 + idx[1]]*180/np.pi))
                            plt.plot(x, y, linestyle=':', color='red')
                            # plt.plot(x[1:-1], y[1:-1], marker='.',
                            #          linestyle='None', color='red')
                            plt.plot(x[0], y[0], marker=m1,
                                     linestyle='None', color='red', markersize=(markersize*scale)**2)
                            plt.plot(x[-1], y[-1], marker='x',
                                     linestyle='None', color='red')

                            print('o {}'.format(seg[0, 15 + idx[1]]*180/np.pi))
                            print('o2 {}'.format(seg[0, 8 + idx[1]]*180/np.pi))
                            print('o3 {}'.format(
                                seg[0, 15 + idx[1]]*180/np.pi))
                            print('o4 {}'.format(angle_to_pipi(
                                seg[0, 15 + idx[1]] - seg[0, 8 + idx[1]]) * 180 / np.pi))

                            plt.plot(xo, yo, linestyle=':', color='k')
                            # plt.plot(x[1:-1], y[1:-1], marker='.',
                            #          linestyle='None', color='k')
                            m1, scale = gen_arrow_head_marker(
                                seg[0, 15 + idx[1]]
                            )
                            markersize = 5

                            plt.plot(xo[0], yo[0], marker=m1, markersize=(markersize*scale)**2,
                                     linestyle='None', color='k')
                            plt.plot(xo[-1], yo[-1], marker='x',
                                     linestyle='None', color='k')

                            outer = plt.Circle(
                                (0, 0), 0.25, color='blue', fill=False)
                            ax.add_artist(outer)

                        ax.set_xlim([-0.3, 0.3])
                        ax.set_ylim([-0.3, 0.3])
                        plt.savefig(
                            path + 'test.png')
                        plt.close()
                        input()


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])[:10]  # !remove
        if len(pos) == 0:
            continue
        data[e] = {'pos': [], 'vel': [], 'dist': [], 'dist_to_wall': []}
        for p in pos:
            p = np.loadtxt(p) * args.radius
            v = Velocities([p], args.timestep).get()[0]
            data[e]['pos'].append(p)
            data[e]['vel'].append(v)
            data[e]['dist'].append(np.sqrt(
                (p[:, 0] - p[:, 2]) ** 2 + (p[:, 1] - p[:, 3]) ** 2))

            dist_mat = []
            for i in range(p.shape[1] // 2):
                distance = args.radius - \
                    np.sqrt(p[:, i * 2] ** 2 + p[:, i * 2 + 1] ** 2)
                dist_mat.append(distance)
            dist_mat = np.array(dist_mat).T
            data[e]['dist_to_wall'].append(dist_mat)

    future_trajectory_variance(data, path, None, args)


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
    parser.add_argument('--prediction_len', type=int,
                        help='Predictions to plot',
                        required=True)
    parser.add_argument('--observation_len', type=int,
                        help='Observations to plot',
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
    parser.add_argument('--radius_grid_res',
                        type=int,
                        help='Resolution (in m) for the radius of the focal individual in the future trajectory variance plot',
                        default=0.025,
                        required=False)
    parser.add_argument('--angle_grid_res',
                        type=int,
                        help='Resolution (in deg) for the angle to the wall of the focal individual future trajectory variance plot',
                        default=5,
                        required=False)
    parser.add_argument('--interdist_grid_res',
                        type=float,
                        help='Resolution (in m) for the interinidividual distance in the future trajectory variance plot',
                        default=0.025,
                        required=False)
    parser.add_argument('--viewing_angle_grid_res',
                        type=float,
                        help='Resolution (in degr) for the interinidividual distance in the future trajectory variance plot',
                        default=45,
                        required=False)
    parser.add_argument('--center',
                        type=float,
                        nargs='+',
                        help='The centroidal coordinates for the setups used',
                        default=[0.0, 0.0],
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
