#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities
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


def vplot(data, ax, args, palette=['#1e81b0', '#D61A3C', '#48A14D'], ticks=False, orient='v'):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 12}
    sns.set_context("paper", rc=paper_rc)

    ax = sns.violinplot(data=data, width=0.4, notch=False,
                        saturation=1, linewidth=1.0, ax=ax,
                        #  whis=[5, 95],
                        cut=0,
                        # inner='quartile',
                        orient=orient,
                        gridsize=args.kde_gridsize,
                        showfliers=False,
                        palette=palette
                        )
    means = []
    stds = []
    # for d in data:
    #     means.append([np.nanmean(list(d))])
    #     stds.append([np.nanstd(list(d))])
    # sns.swarmplot(data=means, palette=['#000000'] * 10,
    #               marker='H', size=5, ax=ax)
    return ax, means, stds


def bplot(data, ax, args, palette=['#1e81b0', '#D61A3C', '#48A14D'], ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    ax = sns.boxplot(data=data, width=0.25, notch=False,
                     saturation=1, linewidth=1.0, ax=ax,
                     #  whis=[5, 95],
                     showfliers=False,
                     palette=palette
                     )

    means = []
    stds = []
    # for d in data:
    #     means.append([np.nanmean(list(d))])
    #     stds.append([np.nanstd(list(d))])
    # sns.swarmplot(data=means, palette=['#000000'] * 10,
    #               marker='H', size=5, ax=ax)
    return ax, means, stds


def cdist_plots(data, path, ax, args, orient='v', palette=['#1e81b0', '#D61A3C', '#48A14D']):
    dists = {}
    for e in reversed(data.keys()):
        dists[e] = []

        l = [[], []]
        for sl in data[e]['cdist']:
            l[0] += [sl[0]]
            l[1] += [sl[1]]
        dists[e] = l

        print(e)
        print('Avg ID 0: {}'.format(np.mean(l[0])))
        print('Avg remaining ID: {}'.format(np.mean(l[1])))

    npalette = palette
    if 'Disc-shaped' in data.keys() and 'Biomimetic' in data.keys() and 'Fish' not in data.keys():
        npalette = [palette[2], palette[1]]

    if 'Disc-shaped' not in data.keys() and 'Biomimetic' in data.keys() and 'Fish' in data.keys():
        npalette = [palette[0], palette[1]]

    if 'Biomimetic' not in data.keys() and 'Disc-shaped' in data.keys() and 'Fish' in data.keys():
        npalette = [palette[2], palette[0]]

    if 'Disc-shaped' in data.keys() and 'Biomimetic' in data.keys() and 'Fish' in data.keys():
        npalette = [palette[0], palette[2], palette[1]]

    for idx, e in enumerate(reversed(dists.keys())):
        print('seq', e)
        if 'Fish' in e:
            npalette = [palette[0]]
        if 'Biomimetic' in e:
            npalette = [palette[1], palette[0]]

        ax[idx], m, s = vplot(dists[e], ax[idx], args,
                              orient=orient, palette=npalette)
        if orient == 'h':
            ax[idx].set_xlabel(r'$d_n$ ($cm$)', fontsize=11)
            ax[idx].set_ylabel(r'PDF', fontsize=11)
        else:
            ax[idx].set_ylabel(r'$d_n$ ($cm$)', fontsize=11)
            ax[idx].set_xlabel(r'PDF', fontsize=11)

    return ax


def idist_plots(data, path, ax, args, orient='v', palette=['#1e81b0', '#D61A3C', '#48A14D']):
    dists = []
    for e in reversed(data.keys()):
        l = []
        for sl in data[e]['idist']:
            l += sl
        dists.append(sl)

    npalette = palette
    if len(dists) == 5:
        if 'Biomimetic' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[1], palette[0]]
        else:
            npalette = [palette[0]] * 5
    else:
        if 'Disc-shaped' in data.keys() and 'Biomimetic' in data.keys() and 'Fish' not in data.keys():
            npalette = [palette[2], palette[1]]

        if 'Disc-shaped' not in data.keys() and 'Biomimetic' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[0], palette[1]]

        if 'Biomimetic' not in data.keys() and 'Disc-shaped' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[2], palette[0]]

        if 'Disc-shaped' in data.keys() and 'Biomimetic' in data.keys() and 'Fish' in data.keys():
            npalette = [palette[0], palette[2], palette[1]]

    ax, m, s = vplot(dists, ax, args, orient=orient, palette=npalette)
    if orient == 'h':
        ax.set_xlabel(r'$d_i$ ($cm$)', fontsize=11)
        ax.set_ylabel('PDF')
    else:
        ax.set_ylabel(r'$d_i$ ($cm$)', fontsize=11)
        ax.set_xlabel('PDF')
    return ax


def plot(exp_files, path, args, palette=['#1e81b0', '#D61A3C', '#48A14D']):
    data = {}
    num_inds = -1

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
        data[e]['idist'] = []
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

            samples += positions.shape[0]
            num_inds = positions.shape[1] // 2
            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

            if num_inds >= 2:
                dist = np.zeros((1, positions.shape[0]))
                for i in range(1, num_inds):
                    dist += (positions[:, 0] - positions[:, i*2]) ** 2 + \
                        (positions[:, 1] - positions[:, i*2 + 1]) ** 2
                interind_dist = np.sqrt(dist / (num_inds - 1))
                interind_dist = interind_dist.tolist()

            # elif num_inds == 2:
            #     interind_dist = np.sqrt(
            #         (positions[:, 0] - positions[:, 2]) ** 2 + (positions[:, 1] - positions[:, 3]) ** 2).tolist()
            else:
                assert False, 'Unsupported number of individuals'

            data[e]['idist'].append(interind_dist)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
        print('{} has {} samples'.format(e, samples))

    _ = plt.figure(figsize=(6, 8))
    ax = plt.gca()

    idist_plots(data, path, ax, args, palette=palette)

    plt.savefig(path + 'idist_plots.png')
