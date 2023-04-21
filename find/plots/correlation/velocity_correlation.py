#!/usr/bin/env python
import glob
import argparse
from tqdm import tqdm

from find.utils.features import Velocities
from find.utils.utils import compute_leadership
from find.plots.common import *


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
                cor0 = data[0][it, 1] * data[0][itp, 1] + \
                    data[0][it, 0] * data[0][itp, 0]
                cor_l[itcor] += cor0

                cor0 = data[1][it, 1] * data[1][itp, 1] + \
                    data[1][it, 0] * data[1][itp, 0]
                cor_f[itcor] += cor0

                ndata[itcor] += 1

    return (cor_l, cor_f), ndata


def corv(data, ax, args):
    lines = ['-', '--', ':']
    linecycler = cycle(lines)
    new_palette = uni_palette()
    # new_palette *= 3
    colorcycler = cycle(sns.color_palette(new_palette))

    if not args.robot:
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

            if 'Simu' in k:
                dtcor = args.ntcor * args.bt
            else:
                dtcor = args.ntcor * args.timestep
            ntcorsup = int(args.tcor / dtcor)

            cor_l = np.zeros(shape=(ntcorsup, 1))
            cor_f = np.zeros(shape=(ntcorsup, 1))
            ndata = np.ones(shape=(ntcorsup, 1))

            for i in tqdm(range(len(velocities)), desc='Processing {}'.format(k), leave=True):
                c, n = compute_correlation(
                    (leader_velocities[i], follower_velocities[i]), args.tcor, args.ntcor, dtcor, ntcorsup, args)
                cor_l += c[0]
                cor_f += c[1]
                ndata += n

            if k == 'Robot':
                time = np.array(range(ntcorsup)) * 0.1
            else:
                time = np.array(range(ntcorsup)) * args.timestep

            ccolour = next(colorcycler)
            ts = (cor_l + cor_f) / (2*ndata)
            ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=ccolour,
                              linestyle=next(linecycler), label=k)
            ts = cor_l / ndata
            ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=ccolour,
                              linestyle=next(linecycler), label='Leader (' + k + ')')
            ts = cor_f / ndata
            ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=ccolour,
                              linestyle=next(linecycler), label='Follower (' + k + ')')
    else:
        for k in sorted(data.keys()):
            ridcs = data[k]['ridx']
            velocities = data[k]['vel']
            robot_velocities = []
            neigh_velocities = []

            for idx in range(len(velocities)):
                vel = velocities[idx]
                num_inds = vel.shape[1] // 2
                ridx = ridcs[idx]
                if ridx < 0:
                    ridx = 0

                rsorted_vel = vel[:, ridx*2:(ridx*2+2)]
                for nidx in range(num_inds):
                    if nidx == num_inds:
                        continue
                    rsorted_vel = np.hstack(
                        [rsorted_vel, vel[:, nidx*2:(nidx*2+2)]])

                robot_velocities.append(rsorted_vel[:, :2])
                neigh_velocities.append(rsorted_vel[:, 2:])

            if 'Simu' in k:
                dtcor = args.ntcor * args.bt
            else:
                dtcor = args.ntcor * args.timestep
            ntcorsup = int(args.tcor / dtcor)

            cor_rob = np.zeros(shape=(ntcorsup, 1))
            cor_neigh = np.zeros(shape=(ntcorsup, 1))
            ndata = np.ones(shape=(ntcorsup, 1))

            for i in tqdm(range(len(velocities)), desc='Processing {}'.format(k)):
                c, n = compute_correlation(
                    (robot_velocities[i], neigh_velocities[i]), args.tcor, args.ntcor, dtcor, ntcorsup, args)
                cor_rob += c[0]
                cor_neigh += c[1]
                ndata += n

            if 'Simu' in k:
                time = np.array(range(ntcorsup)) * args.bt
            else:
                time = np.array(range(ntcorsup)) * args.timestep

            ccolour = next(colorcycler)
            ts = (cor_rob + cor_neigh) / (2*ndata)
            ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=ccolour,
                              linestyle=next(linecycler), label=k)
            ts = cor_rob / ndata
            ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=ccolour,
                              linestyle=next(linecycler), label='Robot (' + k + ')')
            ts = cor_neigh / ndata
            ax = sns.lineplot(x=time.tolist(), y=ts.T.tolist()[0], ax=ax, color=ccolour,
                              linestyle=next(linecycler), label='Neigh (' + k + ')')

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
    for t in args.type:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    plot(exp_files, './', args)
