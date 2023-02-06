#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities, Accelerations
from find.utils.utils import compute_leadership
from find.plots.common import *

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


oc = mpl.path.Path([(0, 0), (1, 0)])

handles_a = [
    mlines.Line2D([0], [0], color='black', marker=oc,
                  markersize=6, label='Median'),
    mlines.Line2D([], [], linestyle='none', color='black', marker='H',
                  markersize=4, label='Mean'),
    # mlines.Line2D([], [], linestyle='none', markeredgewidth=1, marker='o',
    #               color='black', markeredgecolor='w', markerfacecolor='black', alpha=0.6,
    #               markersize=5, label='Sample'),
    # mlines.Line2D([], [], linestyle='none', markeredgewidth=0, marker='*',
    #               color='black', markeredgecolor='w', markerfacecolor='black',
    #               markersize=6, label='Statistical significance'),
]


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def vplot(data, ax, args, ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    ax = sns.violinplot(data=data, width=0.25, notch=False,
                        saturation=1, linewidth=1.0, ax=ax,
                        #  whis=[5, 95],
                        cut=0,
                        # inner='quartile',
                        gridsize=args.kde_gridsize,
                        showfliers=False,
                        palette=["#ed8b02", "#e74c3c"]
                        )
    means = []
    stds = []
    # for d in data:
    #     means.append([np.nanmean(list(d))])
    #     stds.append([np.nanstd(list(d))])
    # sns.swarmplot(data=means, palette=['#000000'] * 10,
    #               marker='H', size=5, ax=ax)
    return ax, means, stds


def bplot(data, ax, ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    ax = sns.boxplot(data=data, width=0.25, notch=False,
                     saturation=1, linewidth=1.0, ax=ax,
                     #  whis=[5, 95],
                     showfliers=False,
                     palette=["#ed8b02", "#e74c3c"]
                     )

    means = []
    stds = []
    # for d in data:
    #     means.append([np.nanmean(list(d))])
    #     stds.append([np.nanstd(list(d))])
    # sns.swarmplot(data=means, palette=['#000000'] * 10,
    #               marker='H', size=5, ax=ax)
    return ax, means, stds


def acc_plots(data, path, ax, args):
    if not args.separate:
        dists = []
        for e in data.keys():
            l = []
            for sl in data[e]['racc']:
                l += sl.tolist()
            dists.append(sl)
        ax, m, s = vplot(dists, ax, args)
        return ax
    else:
        for ne, e in enumerate(data.keys()):
            print(e)

            num_inds = data[e]['pos'][0].shape[1] // 2
            dists = [[] for _ in range(num_inds)]
            ids = [[] for _ in range(num_inds)]

            for nvec, vecs in enumerate(data[e]['racc']):
                order = list(range(num_inds))
                ridx = data[e]['ridx'][nvec]
                if ridx >= 0:
                    order.remove(ridx)
                    order = [ridx] + order

                for no, idx in enumerate(order):
                    dists[no] += vecs[:, idx].tolist()
                    if args.robot and ridx == idx:
                        ids[no] = 'Robot/Lure'
                    else:
                        ids[no] = 'Fish'

            ax[ne], m, s = vplot(dists, ax[ne], args)
            ax[ne].set_xticklabels(ids)
            ax[ne].set_title(e)
            if ne == 0:
                ax[ne].set_ylabel(r'Acceleration ($cm/s^2$)')

        return ax


def plot(exp_files, path, args):
    data = {}
    for e in sorted(exp_files.keys()):
        samples = 0

        if e == 'BOBI':
            timestep = args.bt
        elif e == 'F44':
            timestep = args.f44t
        else:
            timestep = args.timestep

        pos = glob.glob(args.path + '/' + exp_files[e])
        if len(pos) == 0:
            continue
        data[e] = {}
        data[e]['pos'] = []
        data[e]['vel'] = []
        data[e]['racc'] = []
        if args.robot:
            data[e]['ridx'] = []

        for p in pos:
            if e == 'Virtual (Toulouse)':
                f = open(p)
                # to allow for loading fortran's doubles
                strarray = f.read().replace("D+", "E+").replace("D-", "E-")
                f.close()
                num_ind = len(strarray.split('\n')[0].strip().split('  '))
                positions = np.fromstring(
                    strarray, sep='\n').reshape(-1, num_ind) * args.radius
            elif e == 'Virtual (Toulouse cpp)':
                positions = np.loadtxt(p)[:, 2:] * args.radius
            else:
                positions = np.loadtxt(p) * args.radius
            velocities = Velocities([positions], timestep).get()[0]
            accelerations = Accelerations([velocities], timestep).get()[0]

            samples += positions.shape[0]
            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

            tup = []
            for i in range(accelerations.shape[1] // 2):
                linear_acceleration = np.sqrt(
                    accelerations[:, i * 2] ** 2 + accelerations[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_acceleration)
            data[e]['racc'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
        print('{} has {} samples'.format(e, samples))

    if args.separate:
        plt.figure(figsize=(6, 8))
        fig, ax = plt.subplots(1, 2, sharey='row')

        ax = acc_plots(data, path, ax, args)

        plt.savefig(path + 'acc_plots_sep.png'.format(e))

    else:
        _ = plt.figure(figsize=(6, 8))
        ax = plt.gca()

        ax = acc_plots(data, path, args)

        ax.set_xticklabels(labels)
        plt.savefig(path + 'acc_plots.png')
