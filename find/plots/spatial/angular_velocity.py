#!/usr/bin/env python
import glob
import argparse

from find.plots.common import *
from find.utils.features import Velocities
from find.utils.utils import angle_to_pipi


def plot(exp_files, path, args):
    ccycler = uni_cycler()
    linecycler = uni_linecycler()
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = []
        for p in pos:
            matrix = np.loadtxt(p) * args.radius
            matrix = Velocities([matrix], args.timestep).get()[0]
            for i in range(matrix.shape[1] // 2):
                angles = np.arctan2(matrix[:, i * 2 + 1], matrix[:, i * 2])
                data[e].append(angles)

    labels = []
    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    for i, k in enumerate(sorted(data.keys())):
        vectors = data[k]
        labels.append(k)
        cvector = []
        for v in vectors:
            phis = v[1:]
            phis_tm1 = v[:-1]
            phis = list(map(angle_to_pipi, phis - phis_tm1))
            cvector += phis
        cvector = list(map(lambda x: x * 180 / np.pi, cvector))
        sns.kdeplot(cvector, ax=ax,
                    color=next(ccycler), linestyle=next(linecycler), linewidth=uni_linewidth, label=k, gridsize=args.kde_gridsize)

    ax.set_xlabel('Angular change between successive timesteps (degrees)')
    ax.set_ylabel('PDF')
    ax.legend()
    plt.savefig(path + 'angular_velocity.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Angular velocity histogram figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Timestep',
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
