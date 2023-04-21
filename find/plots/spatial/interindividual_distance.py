#!/usr/bin/env python
import glob
import argparse

from find.plots.common import *


def ci(data, key, ax, args, clipping_range=[0.0, 0.6]):
    ccycler = uni_cycler()

    for i, k in enumerate(sorted(data.keys())):
        ccolor = next(ccycler)

        if k == 'path':
            continue

        cis = data[k][key]
        ridcs = data[k]['ridx']
        num_inds = data[k]['pos'][0].shape[1] // 2

        rdist = []
        ndist = []

        c_rob = []
        c_fish = []
        c_all = []
        for idx in range(len(cis)):
            ridx = ridcs[idx]
            cr = cis[idx][ridx]

            cf = []
            for i in range(num_inds):
                if ridx >= 0 and ridx == i:
                    continue
                cf += cis[idx][i]

            c_rob += cr
            c_fish += cf
            c_all += cr + cf

        if ridx >= 0:
            print('Fish-only {}'.format(key), k)
            print('LF: ', np.mean(c_fish), np.std(c_fish))
            ax = sns.kdeplot(c_fish, ax=ax,
                             color=ccolor, linestyle='--', linewidth=uni_linewidth, label=k, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.3, cut=0)

            print('Robot only {}'.format(key), k)
            print('LF: ', np.mean(c_rob), np.std(c_rob))
            ax = sns.kdeplot(c_rob, ax=ax,
                             color=ccolor, linestyle=':', linewidth=uni_linewidth, label=k, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.3, cut=0)

            print('All {}'.format(key), k)
            print('LF: ', np.mean(c_all), np.std(c_all))
            ax = sns.kdeplot(c_all, ax=ax,
                             color=ccolor, linestyle='-', linewidth=uni_linewidth, label=k, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.3, cut=0)

        else:
            print('Fish-only {}'.format(key), k)
            print('LF: ', np.mean(c_fish), np.std(c_fish))
            ax = sns.kdeplot(c_fish, ax=ax,
                             color=ccolor, linestyle='-', linewidth=uni_linewidth, label=k, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.3, cut=0)
    return ax


def interindividual_distance(data, ax, args, clipping_range=[0.0, 0.5]):
    lines = ['-']
    linecycler = cycle(lines)
    ccycler = uni_cycler()
    for i, k in enumerate(sorted(data.keys())):
        if k == 'path':
            continue

        vectors = data[k]
        cvector = []
        for v in vectors:
            # cvector += v.tolist()
            cvector += v

        print('Interindividual', k)
        print('LF: ', np.mean(cvector),
              np.std(cvector))
        ax = sns.kdeplot(cvector, ax=ax,
                         color=next(ccycler), linestyle=next(linecycler), linewidth=uni_linewidth, label=k, gridsize=args.kde_gridsize, clip=clipping_range, bw_adjust=0.3, cut=0)
        if 'path' in list(data.keys()):
            np.savetxt(data['path'] + '/dist_di_{}.dat'.format(k), cvector)
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
    for t in args.type:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
