#!/usr/bin/env python
import glob
import argparse
from itertools import groupby
from operator import itemgetter

from find.utils.utils import angle_to_pipi, compute_leadership
from find.utils.features import Velocities
from find.plots.common import *


def plot_distribution(data, ax, args):
    # lines = ['-', '--', ':']
    linecycler = cycle(uni_lines)
    new_palette = uni_palette()
    # for p in uni_palette():
    #     new_palette.extend([p, p, p])
    ccycler = cycle(sns.color_palette(new_palette))

    sample_count = {}
    leadership = {}
    for k in sorted(data.keys()):
        sample_count[k] = 0
        pos = data[k]['pos']
        vel = data[k]['vel']
        leadership[k] = []
        for idx in range(len(pos)):
            sample_count[k] += pos[idx].shape[0]
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])
            leadership[k].append(leadership_timeseries)

    for kidx, k in enumerate(sorted(data.keys())):
        leaders = leadership[k]

        occurences = {}
        for e in range(len(data[k]['pos'])):
            lts = np.array(leaders[e])[:, 1].tolist()

            cons_count = 1
            for i in range(len(lts) - 1):
                if lts[i] != lts[i+1]:
                    if cons_count not in occurences.keys():
                        occurences[cons_count] = 1.
                    else:
                        occurences[cons_count] += 1.
                    cons_count = 1
                else:
                    cons_count += 1

        dist = np.array([list(occurences.keys()), list(occurences.values())]).T

        print(np.mean(dist[:, 0]) * args.timestep, k)
        col = next(ccycler)
        ax = sns.kdeplot(dist[:, 1], ax=ax, color=col,
                         linestyle=next(linecycler), label=k, linewidth=uni_linewidth, gridsize=args.kde_gridsize,
                         clip=[0, 500],
                         bw_adjust=.2
                         )

        # ticks = np.arange(0, 501, 50)
        # ax.set_xticks(ticks)
        # time = ticks * args.timestep
        # ax.set_xticklabels(time)
        # ax.axvline(np.mean(dist[:, 0]), color=col, linestyle='dashed')
        # ax.text(np.mean(dist[:, 0]) * 1.1, 0.018 - kidx * 0.003,
        #         'Mean: {:.2f}'.format(np.mean(dist[:, 0]) * args.timestep), color=col)
        # ax.set_xlim([-0.1, 500])
    return ax


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

    # relative angle to neigh
    _ = plt.figure(figsize=(6, 5))
    ax = plt.gca()

    ax = plot_distribution(data, ax, args)

    ax.set_xlabel(r'$t$ (s)')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig(path + 'leadership_distribution.png')


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
    for t in args.types:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
