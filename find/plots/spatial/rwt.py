#!/usr/bin/env python
import os
import glob
import argparse

from find.utils.utils import angle_to_pipi, compute_leadership
from find.utils.features import Velocities
from find.plots.common import *

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)


def rwt_plot(data, ax, args):
    lines = ['--', ':']
    linecycler = cycle(lines)
    new_palette = uni_palette()
    new_palette *= 3
    colorcycler = cycle(sns.color_palette(new_palette))

    label_size = 10
    for axc in ax.flat:
        axc.tick_params(axis='both', labelsize=label_size)
        # axc.set_xlabel('x-axis label', fontsize=legend_size)
        # axc.set_ylabel('y-axis label', fontsize=legend_size)
        # axc.set_title('Title', fontsize=title_size, color="C0")

    ax[-1, 0].set_xlabel(r'$t$ (s)')
    ax[-1, 1].set_xlabel(r'$t$ (s)')

    for i, k in enumerate(data.keys()):
        rws = data[k]['distance_to_wall']
        for j, rw in enumerate(rws):
            t = np.arange(1, rw.shape[0] + 1, 1) * 0.12

            ax[j, 0].set_ylabel(r'$r_\mathrm{w}(t)$ (m)')
            # ax[0, 1].set_ylabel(r'$r_\mathrm{w}(t)$ (m)')

            ax[j, 0].set_xlim([0, 6000])
            ax[j, 1].set_xlim([0, 6000])
            lw = 0.5
            if np.min(rw[:, 0]) < -1:
                ax[j, 0].set_ylim([-250, 10])
                ax[j, 0].set_yticks(np.arange(-250, 1, 250))
                lw = 1.0
            else:
                ax[j, 0].set_ylim([-0.25, 0.25])

            if np.min(rw[:, 1]) < -1:
                ax[j, 1].set_ylim([-250, 10])
                ax[j, 1].set_yticks(np.arange(-250, 1, 250))
                lw = 1.0
            else:
                ax[j, 1].set_ylim([-0.25, 0.25])

            sns.lineplot(y=rw[:, 0], x=t, ax=ax[j, 0], linewidth=lw)
            sns.lineplot(y=rw[:, 1], x=t, ax=ax[j, 1], linewidth=lw)

    return ax


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        count = 0

        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['distance_to_wall'] = []
        data[e]['pos'] = []
        if args.robot:
            data[e]['ridx'] = []

        for num, p in enumerate(pos):
            matrix = np.loadtxt(p) * args.radius
            count += matrix.shape[0]
            dist_mat = []

            for i in range(matrix.shape[1] // 2):
                distance = args.radius - \
                    np.sqrt(matrix[:, i * 2] ** 2 + matrix[:, i * 2 + 1] ** 2)
                dist_mat.append(distance)
            dist_mat = np.array(dist_mat).T

            thres0 = 0.5
            etime0 = dist_mat.shape[0]
            flag = False
            # print(np.min(dist_mat))
            for i in range(dist_mat.shape[0]):
                for j in range(dist_mat.shape[1]):
                    if dist_mat[i, j] < -thres0:
                        etime0 = i
                        flag = True
                if flag:
                    break

            thres1 = 1.0
            etime1 = dist_mat.shape[0]
            flag = False
            # print(np.min(dist_mat))
            for i in range(dist_mat.shape[0]):
                for j in range(dist_mat.shape[1]):
                    if dist_mat[i, j] < -thres1:
                        etime1 = i
                        flag = True
                if flag:
                    break

            thres2 = 3.0
            etime2 = dist_mat.shape[0]
            flag = False
            # print(np.min(dist_mat))
            for i in range(dist_mat.shape[0]):
                for j in range(dist_mat.shape[1]):
                    if dist_mat[i, j] < -thres2:
                        etime2 = i
                        flag = True
                if flag:
                    break

            thres3 = 10.0
            etime3 = dist_mat.shape[0]
            flag = False
            # print(np.min(dist_mat))
            for i in range(dist_mat.shape[0]):
                for j in range(dist_mat.shape[1]):
                    if dist_mat[i, j] < -thres3:
                        etime3 = i
                        flag = True
                if flag:
                    break

            print(etime0 * 0.12, etime1 * 0.12, etime2 * 0.12, etime3 * 0.12)

            data[e]['distance_to_wall'].append(dist_mat)
            data[e]['pos'].append(matrix)

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                if os.path.exists(r):
                    ridx = np.loadtxt(r).astype(int)
                else:
                    ridx = -1
                data[e]['ridx'].append(int(ridx))

        print('Total samples: {}'.format(count))

    _, ax = plt.subplots(figsize=(8, 10),
                         nrows=10, ncols=2,
                         gridspec_kw={
        'wspace': 0.3, 'hspace': 1.3}
    )
    rwt_plot(data, ax, args)

    ax[0, 0].set_title('DLI agent 0')
    ax[0, 1].set_title('DLI agent 1')
    # ax.set_xlabel(r'$r_w$')
    # ax.set_ylabel('PDF')
    # ax.legend()
    plt.savefig(path + 'rwt.png')


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
