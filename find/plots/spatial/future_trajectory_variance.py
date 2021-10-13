#!/usr/bin/env python
import glob
import argparse
from find.plots.spatial.relative_orientation import viewing_angle

from find.utils.utils import angle_to_pipi
from find.utils.features import Velocities
from find.plots.common import *

import pandas as pd

def compute_grid(data, args):
    grid = {}
    for k in sorted(data.keys()):
        grid[k] = {}
        skipped_count = 0
        total_count = 0

        trajectory_segments = []
        segment_len = args.observation_len + args.prediction_len

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
                break # not enough samples to generate the plot
 
            # headings
            hdgs = np.empty((p.shape[0], 0))
            for i in range(p.shape[1] // 2):
                hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

            # angles to the wall
            angles_to_wall = np.empty((hdgs.shape[0], hdgs.shape[1]))
            for i in range(angles_to_wall.shape[1] // 2):
                angles_to_wall[:, i] = hdgs[:, i] - np.arctan2(p[:, i*2+1], p[:, i*2])

            # viewing angles
            angle_dif_focal = hdgs[:, 0] - \
                np.arctan2(p[:, 3] - p[:, 1], p[:, 2] - p[:, 0])
            angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))

            angle_dif_neigh = hdgs[:, 1] - \
                np.arctan2(p[:, 1] - p[:, 3], p[:, 0] - p[:, 2])
            angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))

            viewing_angles = np.array([angle_dif_focal, angle_dif_neigh]).T

            # complete matrix with info about the trajectories of the neighbouring fish
            matrix = np.hstack((p, v, angles_to_wall, dwall, idist.reshape(-1, 1), viewing_angles))
            for i in range(0, matrix.shape[0], segment_len):
                # X1 Y1 X2 Y2 VX1 VY1 VX2 VY2 Th1 Th2 D1 D2 Idist Psi1 Psi2
                # 0  1  2  3  4   5   6   7   8   9   10 11 12    13   14
                trajectory_segments.append(matrix[i:(i+segment_len), :])

        # --- generate grids

        # radii grid
        for ts_idx, ts in enumerate(trajectory_segments): 
            for i in range(1, len(r_grid)):
                if r_grid[i-1] <= ts[args.observation_len-1, 10] and ts[args.observation_len-1, 10] < r_grid[i]:
                    idx_dict[(r_grid[i-1], r_grid[i])]['all'].append((ts_idx, 0))
                if r_grid[i-1] <= ts[args.observation_len-1, 11] and ts[args.observation_len-1, 11] < r_grid[i]:
                    idx_dict[(r_grid[i-1], r_grid[i])]['all'].append((ts_idx, 1))

        # radii and angle to the wall
        for dgrid, sub_dict in idx_dict.items():
            for idx in sub_dict['all']:
                for i in range(1, len(a_grid)):
                    ts = trajectory_segments[idx[0]]

                    if (a_grid[i-1], a_grid[i]) not in sub_dict.keys():
                        idx_dict[dgrid][(a_grid[i-1], a_grid[i])] = {}
                        idx_dict[dgrid][(a_grid[i-1], a_grid[i])]['all'] = [] # ! this could be moved to not create empty lists if there are no idcs in it

                    if a_grid[i-1] <= np.abs(ts[args.observation_len-1, 8 + idx[1]] * 180 / np.pi) and np.abs(ts[args.observation_len-1, 8 + idx[1]] * 180 / np.pi) < a_grid[i]:
                        idx_dict[dgrid][(a_grid[i-1], a_grid[i])]['all'].append(idx)

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
                            idx_dict[dgrid][agrid][(idist_grid[i-1], idist_grid[i])] = {}
                            idx_dict[dgrid][agrid][(idist_grid[i-1], idist_grid[i])]['all'] = [] # ! this could be moved to not create empty lists if there are no idcs in it

                        if idist_grid[i-1] <= ts[args.observation_len-1, 12] and ts[args.observation_len-1, 12] < idist_grid[i]:
                            idx_dict[dgrid][agrid][(idist_grid[i-1], idist_grid[i])]['all'].append(idx)


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
                                va_idx = 14 # we want the viewing angle of the neighbour to the focal
                            else:
                                neigh = 0  
                                va_idx = 13


                            if (psi_grid[i-1], psi_grid[i]) not in sub_dict3.keys():
                                idx_dict[dgrid][agrid][idist_grid][(psi_grid[i-1], psi_grid[i])] = {}
                                idx_dict[dgrid][agrid][idist_grid][(psi_grid[i-1], psi_grid[i])]['all'] = [] # ! this could be moved to not create empty lists if there are no idcs in it
                            
                            if psi_grid[i-1] <= ts[args.observation_len-1, va_idx] * 180 / np.pi and ts[args.observation_len-1, va_idx] * 180 / np.pi < psi_grid[i]:
                                idx_dict[dgrid][agrid][idist_grid][(psi_grid[i-1], psi_grid[i])]['all'].append(idx)

        grid[k]['idx_grid'] = idx_dict
        grid[k]['seg'] = trajectory_segments
        print('{} skipped: {} / {} experiment files'.format(k, skipped_count, total_count))
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
        segment = grid[k]['seg']

        # plot distance barplots
        _ = plt.figure(figsize=(6, 6))
        ax = plt.gca()

        cell_range = []
        count = []
        for dgrid, sub_dict1 in idx_dict.items():
            if dgrid == 'all':
                continue

            lb = '{:.3f}'.format(round(dgrid[0], 3))
            ub = '{:.3f}'.format(round(dgrid[1], 3))
            cell_range.append('[{}, {}]'.format(lb, ub))
            count.append(len(sub_dict1['all']))

        sns.barplot(x=cell_range, y=count, ax=ax, color=next(ccycler))
        ax.set_xlabel('Radius range (m)')
        ax.set_ylabel('Number of trajectories')
        ax.set_title(k)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path + 'dist_barplot_{}-step_{}'.format(str(args.radius_grid_res).replace('.', '_'), k))

        # plot angles to the wall barplots
        for dgrid, sub_dict1 in idx_dict.items():
            if dgrid == 'all':
                continue

            cell_range = []
            angles_rad = []
            count = []

            for agrid, sub_dict2 in sub_dict1.items():
                if agrid == 'all':
                    continue

                cell_range.append('[{}, {}]'.format(agrid[0], agrid[1]))
                angles_rad.append(agrid[1] * np.pi / 180.)
                count.append(len(sub_dict2['all']))

            _ = plt.figure(figsize=(6, 6))
            ax = plt.gca()
            ccycler = cycle(sns.color_palette(new_palette))

            ax = plt.subplot(projection='polar')
            ax.set_thetamin(0)
            ax.set_thetamax(180)

            ax.bar(angles_rad, count, width=args.angle_grid_res * np.pi / 180., bottom=0.0, color=next(ccycler), alpha=1.0)
            ax.set_xlabel('Radius range (m)')
            ax.set_ylabel('Number of trajectories')
            ax.set_title(k)
            plt.tight_layout()

            lb = '{:.3f}'.format(round(dgrid[0], 3))
            ub = '{:.3f}'.format(round(dgrid[1], 3))
            dgrid_str = '{}_{}'.format(lb.replace('.', '_'), ub.replace('.', '_'))

            plt.savefig(path + 'theta1_pplot_{}-step_{}-radius_{}'.format(str(args.radius_grid_res).replace('.', '_'), dgrid_str, k))
            plt.close()



def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
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
