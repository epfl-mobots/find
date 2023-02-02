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


def vplot(data, ax, ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    ax = sns.violinplot(data=data, width=0.25, notch=False,
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


def vel_plots(data, path, args):
    if not args.separate:
        _ = plt.figure(figsize=(6, 8))
        ax = plt.gca()

        dists = []
        labels = []
        for e in data.keys():
            l = []
            for sl in data[e]['rvel']:
                l += sl.tolist()
            dists.append(sl)
            labels.append(e)
        ax, m, s = vplot(dists, ax)

        ax.set_xticklabels(labels)
        ax.legend()
        plt.savefig(path + 'vel_plots.png')
    else:
        _ = plt.figure(figsize=(6, 8))
        ax = plt.gca()

        for ne, e in enumerate(data.keys()):
            dists = []
            ids = []
            num_inds = data[e]['pos'][0].shape[1] // 2

            if args.robot:
                pass
            #     order = list(range(num_inds))
            #     ridx = data[e]['ridx'][ne]
            #     if ridx >= 0:
            #         order.remove(ridx)
            #         order = [ridx] + order

            #     sl = data[e]['rvel']
            #     for idx in order:
            #         dists.append(sl[idx].tolist())
            #         ids.append(idx)

            # ax, m, s = vplot(dists, ax)
            # ax.legend()
            # plt.savefig(path + 'vel_plots-{}.png'.format(e))

        # m, s = bplot(data[2:4], ax1)
        # ax1.set_ylabel('')
        # ax1.set_xlabel('')
        # ax1.set_title('0.24 s')
        # ax1.set_ylim([0, 1.4])
        # ax1.set_yticklabels([])
        # ax1.set_xticklabels([])
        # ax1.spines['right'].set_color('none')
        # ax1.spines['left'].set_color('none')
        # means.append(m)
        # stds.append(s)

        # m, s = bplot(data[4:], ax2)
        # ax2.set_ylabel('')
        # ax2.set_xlabel('')
        # ax2.set_title('0.36 s')
        # ax2.set_ylim([0, 1.4])
        # ax2.set_yticklabels([])
        # ax2.set_xticklabels([])
        # ax2.spines['left'].set_color('none')
        # means.append(m)
        # stds.append(s)

        # ax0.axvline(x=1.5, ymin=0.0, ymax=1.0, color='black')
        # ax1.axvline(x=1.5, ymin=0.0, ymax=1.0, color='black')

        # ax0.legend(handles=handles_a,
        #            handletextpad=0.5, columnspacing=1,
        #            loc="upper left", ncol=1, framealpha=0, frameon=False, fontsize=9)

        # extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
        #                   edgecolor='none', linewidth=0)
        # shapeList = [
        #     Circle((0, 0), radius=1, facecolor='#ed8b02'),
        #     Circle((0, 0), radius=1, facecolor='#e74c3c'),
        # ]

        # l = fig.legend(shapeList, labels, shadow=True, bbox_to_anchor=(0.5, 0.02),
        #                handletextpad=0.5, columnspacing=1,
        #                loc="lower center", ncol=4, frameon=False, fontsize=9)


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
        data[e]['rvel'] = []
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
            linear_velocity = np.array((velocities.shape[0], 1))

            samples += positions.shape[0]

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

            tup = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)
            data[e]['rvel'].append(np.array(tup).T)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
        print('{} has {} samples'.format(e, samples))

    vel_plots(data, path, args)
