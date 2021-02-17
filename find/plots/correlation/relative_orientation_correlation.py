#!/usr/bin/env python
import glob
import argparse
from tqdm import tqdm

from find.utils.features import Velocities
from find.utils.utils import angle_to_pipi, compute_leadership
from find.plots.common import *

from scipy.stats import norm, rv_histogram


def compute_correlation(data, tcor, ntcor, dtcor, ntcorsup, args):
    cor_l = np.zeros(shape=(ntcorsup, 1))
    cor_f = np.zeros(shape=(ntcorsup, 1))
    ndata = np.ones(shape=(ntcorsup, 1))

    for it in range(data[0].shape[0]):
        if (it+1) % 5000 == 0:
            print('At iteration {} out of {}'.format(it+1, data[0].shape[0]))
        for itcor in range(ntcorsup):
            itp = it + itcor * ntcor
            if (itp < data[0].shape[0]):

                cor0 = np.cos(data[0][it, 0] - data[0][itp, 0])
                cor_l[itcor] += cor0

                cor0 = np.cos(data[1][it, 0] - data[1][itp, 0])
                cor_f[itcor] += cor0

                ndata[itcor] += 1

    return (cor_l, cor_f), ndata


def cortheta(data, ax, args):
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
        relor = data[k]['rel_or']
        lrelor = []
        frelor = []

        for idx in range(len(relor)):
            leadership_mat = np.array(leaders[idx])
            lr = np.copy(relor[idx][:, :2])
            fr = np.copy(relor[idx][:, :2])

            idx_leaders_0 = np.where(leadership_mat[:, 1] == 0)
            idx_leaders_1 = np.where(leadership_mat[:, 1] == 1)

            lr[idx_leaders_0, 0] = relor[idx][idx_leaders_0, 0]
            lr[idx_leaders_1, 1] = relor[idx][idx_leaders_1, 1]

            fr[idx_leaders_0, 0] = relor[idx][idx_leaders_0, 1]
            fr[idx_leaders_1, 1] = relor[idx][idx_leaders_1, 0]

            lrelor.append(lr)
            frelor.append(fr)

        dtcor = args.ntcor * args.timestep
        ntcorsup = int(args.tcor / dtcor)

        cor_l = np.zeros(shape=(ntcorsup, 1))
        cor_f = np.zeros(shape=(ntcorsup, 1))
        ndata = np.ones(shape=(ntcorsup, 1))

        for i in tqdm(range(len(relor)), desc='Processing {}'.format(k)):
            c, n = compute_correlation(
                (lrelor[i], frelor[i]), args.tcor, args.ntcor, dtcor, ntcorsup, args)
            cor_l += c[0]
            cor_f += c[1]
            ndata += n

        time = np.array(range(ntcorsup)) * args.timestep

        ts = (cor_l + cor_f) / (2*ndata)
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
        data[e]['rel_or'] = []
        for p in pos:
            positions = np.loadtxt(p) * args.radius
            velocities = Velocities([positions], args.timestep).get()[0]
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

            hdgs = np.empty((positions.shape[0], 0))
            for i in range(positions.shape[1] // 2):
                hdg = np.arctan2(velocities[:, i*2+1], velocities[:, i*2])
                hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

            # for the focal
            angle_dif_focal = hdgs[:, 0] - \
                np.arctan2(positions[:, 1], positions[:, 0])
            angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))

            # for the neigh
            angle_dif_neigh = hdgs[:, 1] - \
                np.arctan2(positions[:, 3], positions[:, 2])
            angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))

            data[e]['rel_or'].append(
                np.array([angle_dif_focal, angle_dif_neigh]).T)

    _ = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    ax = cortheta(data, ax, args)

    ax.set_xlabel('$t$ (s)')
    ax.set_ylabel(r'$<cos(\theta_w(t) - \theta_w(0))>$')
    ax.legend()
    plt.savefig(path + 'cortheta.png')


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
