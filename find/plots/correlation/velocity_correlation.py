#!/usr/bin/env python
import glob
import argparse
from tqdm import tqdm

from find.utils.features import Velocities
from find.utils.utils import compute_leadership
from find.plots.common import *


def compute_correlation(data, tcor, ntcor, dtcor, ntcorsup, args):
    cor_avg = np.zeros(shape=(ntcorsup, 1))
    cor_l = np.zeros(shape=(ntcorsup, 1))
    cor_f = np.zeros(shape=(ntcorsup, 1))
    ndata = np.ones(shape=(ntcorsup, 1))

    for it in range(data[0].shape[0]):
        if (it+1) % 5000 == 0:
            print('At iteration {} out of {}'.format(it+1, data[0].shape[0]))
        for itcor in range(ntcorsup):
            itp = it + itcor * ntcor
            if (itp < data[0].shape[0]):
                cor0 = data[0][it, 1] * data[0][itp, 1] + \
                    data[0][it, 0] * data[0][itp, 0]
                cor_avg[itcor] += cor0

                cor0 = data[0][it, 3] * data[0][itp, 3] + \
                    data[0][it, 2] * data[0][itp, 2]
                cor_avg[itcor] += cor0

                cor0 = data[1][it, 1] * data[1][itp, 1] + \
                    data[1][it, 0] * data[1][itp, 0]
                cor_l[itcor] += cor0

                cor0 = data[2][it, 1] * data[2][itp, 1] + \
                    data[2][it, 0] * data[2][itp, 0]
                cor_f[itcor] += cor0

                ndata[itcor] += 1

    return (cor_avg, cor_l, cor_f), ndata


def corv(data, ax, args):
    lines = ['-', '--', ':']
    linecycler = cycle(lines)
    new_palette = []
    for p in uni_palette():
        new_palette.extend([p, p, p])
    colorcycler = cycle(sns.color_palette(new_palette))

    leadership = {}
    for k in sorted(data.keys()):
        p = data[k]['pos']
        v = data[k]['vel']
        leadership[k] = []
        for idx in range(len(p)):
            (_, leadership_timeseries) = compute_leadership(p[idx], v[idx])
            leadership[k].append(leadership_timeseries)

    for k in sorted(data.keys()):
        leaders = leadership[k]
        velocities = data[k]['vel']
        leader_velocities = []
        follower_velocities = []

        for idx in range(len(velocities)):
            leadership_mat = np.array(leaders[idx])
            lvel = np.copy(velocities[idx][:, :2])
            fvel = np.copy(velocities[idx][:, 2:])

            idx_leaders_0 = np.where(leadership_mat[:, 1] == 0)
            idx_leaders_1 = np.where(leadership_mat[:, 1] == 1)

            lvel[idx_leaders_0, 0] = velocities[idx][idx_leaders_0, 0]
            lvel[idx_leaders_0, 1] = velocities[idx][idx_leaders_0, 1]
            lvel[idx_leaders_1, 0] = velocities[idx][idx_leaders_1, 2]
            lvel[idx_leaders_1, 1] = velocities[idx][idx_leaders_1, 3]

            fvel[idx_leaders_0, 0] = velocities[idx][idx_leaders_0, 2]
            fvel[idx_leaders_0, 1] = velocities[idx][idx_leaders_0, 3]
            fvel[idx_leaders_1, 0] = velocities[idx][idx_leaders_1, 0]
            fvel[idx_leaders_1, 1] = velocities[idx][idx_leaders_1, 1]

            leader_velocities.append(lvel)
            follower_velocities.append(fvel)

        dtcor = args.ntcor * args.timestep
        ntcorsup = int(args.tcor / dtcor)

        cor_avg = np.zeros(shape=(ntcorsup, 1))
        cor_l = np.zeros(shape=(ntcorsup, 1))
        cor_f = np.zeros(shape=(ntcorsup, 1))
        ndata = np.ones(shape=(ntcorsup, 1))

        for i in tqdm(range(len(velocities)), desc='Processing {}'.format(k), leave=True):
            c, n = compute_correlation(
                (velocities[i], leader_velocities[i], follower_velocities[i]), args.tcor, args.ntcor, dtcor, ntcorsup, args)
            cor_avg += c[0]
            cor_l += c[1]
            cor_f += c[2]
            ndata += n

        time = np.array(range(ntcorsup)) * args.timestep

        ts = cor_avg / (2*ndata)
        ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=next(colorcycler),
                          linestyle=next(linecycler), label=k)
        ts = cor_l / ndata
        ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=next(colorcycler),
                          linestyle=next(linecycler), label='Leader (' + k + ')')
        ts = cor_f / ndata
        ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=next(colorcycler),
                          linestyle=next(linecycler), label='Follower (' + k + ')')
    return ax


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        for p in pos:
            positions = np.loadtxt(p) * args.radius
            velocities = Velocities([positions], args.timestep).get()[0]
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    ax = corv(data, ax, args)

    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel(r'$<V(t) \dot V(0)>$')
    ax.legend()
    plt.savefig(path + 'corv.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Velocity correlation figure')
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
    parser.add_argument('--tcor',
                        type=float,
                        default=25.0,
                        help='Time window to consider when computing correlation metrics',
                        required=False)
    parser.add_argument('--ntcor',
                        type=int,
                        default=1,
                        help='Number of timesteps to includ in the correlation metrics computaion',
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
