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


def vplot(data, ax, args, width=0.4, palette=['#1e81b0', '#D61A3C', '#48A14D'], ticks=False, orient='v'):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 12}
    sns.set_context("paper", rc=paper_rc)

    ax = sns.violinplot(data=data, width=width, notch=False,
                        saturation=1, linewidth=1.0, ax=ax,
                        #  whis=[5, 95],
                        # inner='quartile',
                        # bw=0.35,
                        cut=0,
                        clip=[0, 180],
                        orient=orient,
                        gridsize=args.kde_gridsize,
                        showfliers=False,
                        palette=palette
                        )
    means = []
    stds = []
    for num, d in enumerate(data):
        q50 = np.percentile(d, [50])
        mea = np.mean(d)
        std = np.std(d)
        med = np.median(d)
        ax.scatter(num, q50,
                   zorder=3,
                   color="white",
                   edgecolor='None',
                   s=10)
        print('Mean, std, median: {}, {}, {}'.format(mea, std, med))
        means.append([np.nanmean(list(d))])
        stds.append([np.nanstd(list(d))])
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


def viewing_plots(data, path, ax, args, orient='v', width=0.6, palette=['#1e81b0', '#D61A3C', '#48A14D']):
    if not args.separate:
        psis = []
        for ne, e in enumerate(sorted(data.keys())):
            l = []
            for sl in data[e]['psi']:
                l += sl
            psis.append(l)

        ax, m, s = vplot(psis, ax, args, orient=orient,
                         width=width, palette=palette)

        if orient == 'h':
            ax.set_xlabel(r'$\psi_{ij}$ $(^{\circ})$', fontsize=11)
            ax.set_ylabel(r'PDF', fontsize=11)
        else:
            ax.set_ylabel(r'$\psi_{ij}$ $(^{\circ})$', fontsize=11)
            ax.set_xlabel(r'PDF', fontsize=11)
        return ax
    else:
        pass
        return ax


def relor_neigh_plots(data, path, ax, args, orient='v', width=0.6, palette=['#1e81b0', '#D61A3C', '#48A14D']):
    if not args.separate:
        phis = []
        for ne, e in enumerate(sorted(data.keys())):
            l = []
            for sl in data[e]['phi']:
                l += sl
            phis.append(l)

        ax, m, s = vplot(phis, ax, args, orient=orient,
                         width=width, palette=palette)

        if orient == 'h':
            ax.set_xlabel(r'$\phi_{ij}$ $(^{\circ})$', fontsize=11)
            ax.set_ylabel(r'PDF', fontsize=11)
        else:
            ax.set_ylabel(r'$\phi_{ij}$ $(^{\circ})$', fontsize=11)
            ax.set_xlabel(r'PDF', fontsize=11)
        return ax
    else:
        pass
        return ax


def relor_wall_plots(data, path, ax, args, orient='v', width=0.6, palette=['#1e81b0', '#D61A3C', '#48A14D']):
    if not args.separate:
        thetas = []
        for ne, e in enumerate(sorted(data.keys())):
            l = []
            for sl in data[e]['theta']:
                l += sl
            thetas.append(l)

        ax, m, s = vplot(thetas, ax, args, orient=orient,
                         width=width, palette=palette)

        if orient == 'h':
            ax.set_xlabel(r'$\theta_{\rm w}$ $(^{\circ})$', fontsize=11)
            ax.set_ylabel(r'PDF', fontsize=11)
        else:
            ax.set_ylabel(r'$\theta_{\rm w}$ $(^{\circ})$', fontsize=11)
            ax.set_xlabel(r'PDF', fontsize=11)
        return ax
    else:
        pass
        return ax
