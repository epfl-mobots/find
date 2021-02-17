#!/usr/bin/env python
import glob
import argparse

from find.plots.common import *


def interindividual_distance(data, ax, args):
    ccycler = uni_cycler()
    for i, k in enumerate(sorted(data.keys())):
        vectors = data[k]
        cvector = []
        for v in vectors:
            cvector += v.tolist()

        ax = sns.kdeplot(cvector, ax=ax,
                         color=next(ccycler), linewidth=uni_linewidth, label=k, gridsize=args.kde_gridsize, clip=[0.0, 0.6], bw_adjust=0.3, cut=0)
    return ax


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = []
        for v in pos:
            matrix = np.loadtxt(v) * args.radius
            distance = np.sqrt(
                (matrix[:, 0] - matrix[:, 2]) ** 2 + (matrix[:, 1] - matrix[:, 3]) ** 2)
            data[e].append(distance)

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    interindividual_distance(data, ax, args)

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig(path + 'interindividual_distance.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Interindividual distance figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--radius', '-r', type=float,
                        help='Radius',
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
    for t in args.types:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
