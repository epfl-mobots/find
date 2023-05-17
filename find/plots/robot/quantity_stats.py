#!/usr/bin/env python
import os
import re
import glob
import random
import pickle
import argparse
import numpy as np
from scipy.stats import t

from rich.progress import track, Progress
from rich.console import Console

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, InsetPosition


USE_CI = True
ci = 0.6827

console = Console(color_system='auto')

qax = {
    'velocity': (np.arange(0, 35, 0.5), 0.5),
    'dw': (np.arange(0, 25, 0.5), 0.5),
    'thetaw': (np.arange(0, 180, 1.5), 1.5),
    'psi': (np.arange(-180, 180, 3), 3),
    'd': (np.arange(0, 50, 0.5), 0.5),
    'phi': (np.arange(0, 180, 1.5), 1.5),
    'corx': (None, 1),
    'corv': (None, 1),
    'cortheta': (None, 1),
}

offsets = {
    'velocity': 0.35,
    'dw': 0.6,
    'thetaw': 0.12,
    'psi': 0.023,
    'd': 0.5,
    'phi': 0.1,
    'corx': 0.0,
    'corv': 0.0,
    'cortheta': 0.0,
}


palette = ["#807f7d", "#3498db", "#e74c3c"]
table_data = {}
all_pdfs = {}
SKIP_FOLDERS = ['2_Simu', '1_Experiment', '2_Simu_split']


def annot_axes(ax, xlabel, ylabel, xlim, ylim, xloc, yloc, yscale):
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.xaxis.set_major_locator(MultipleLocator(xloc[0]))
    ax.xaxis.set_minor_locator(MultipleLocator(xloc[1]))
    ax.tick_params(axis="x", which='both',
                   direction="in")
    ax.tick_params(axis="x", which='major', length=2.5, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.0, width=0.7)
    ax.tick_params(axis="y", which='major', length=2.5, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.0, width=0.7)

    ax.set_ylabel(ylabel)
    ylim = [e / yscale for e in ylim]
    ax.set_ylim(ylim)
    ax.get_yaxis().set_major_formatter(
        FuncFormatter(lambda x, p: '{:.1f}'.format(x * yscale, ',')))
    ax.yaxis.set_major_locator(MultipleLocator(yloc[0] / yscale))
    ax.yaxis.set_minor_locator(MultipleLocator(yloc[1] / yscale))
    ax.tick_params(which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.grid(False)
    return ax


def custom_vplot(distributions, quantity, ax, args, half=False, vertical=True, qtype='avg', alpha=0.35, linewidth=1, cpalette=None, linestyle='-'):

    num_distributions = len(distributions[qtype]['mean'])
    if cpalette is None:
        cpalette = palette

    if 'cor' in quantity:
        steps = [0] * num_distributions
    else:
        steps = np.arange(0, num_distributions) * offsets[quantity]

    for i, offset in enumerate(steps):
        # average of individuals
        if not vertical:
            y = distributions[qtype]['mean'][i]
            if 'cor' in quantity:
                if y.shape[0] == 250:  # ! this is hardcoding for timesteps of 0.12s
                    x = np.arange(0, 30, 0.12)
                else:
                    x = np.arange(0, len(y) / 10, 0.1)
            else:
                x = qax[quantity][0]

            fy1 = distributions[qtype]['sd'][i][0]
            fy2 = distributions[qtype]['sd'][i][1]
        else:
            x = distributions[qtype]['mean'][i]
            if 'cor' in quantity:
                if x.shape[0] == 250:  # ! this is hardcoding for timesteps of 0.12s
                    y = np.arange(0, 30, 0.12)
                else:
                    y = np.arange(0, len(x) / 10, 0.1)
                y /= 10  # to make it seconds
            else:
                y = qax[quantity][0]
            fy1 = distributions[qtype]['sd'][i][0]
            fy2 = distributions[qtype]['sd'][i][1]

        if not vertical:
            ax.plot(x, y + offset,
                    color=cpalette[i], linestyle=linestyle, linewidth=linewidth)
            if 'cor' not in quantity and not half:
                ax.plot(
                    x, -y + offset, color=cpalette[i], linestyle=linestyle, linewidth=linewidth)

            ax.fill_between(x, fy1 + offset, fy2 + offset,
                            alpha=alpha, color=cpalette[i])

            if 'cor' not in quantity and not half:
                ax.fill_between(x, -fy1 + offset, -fy2 + offset,
                                alpha=alpha, color=cpalette[i])

        else:
            ax.plot(x + offset, y,
                    color=cpalette[i], linestyle=linestyle, linewidth=linewidth)
            if 'cor' not in quantity and not half:
                ax.plot(-x + offset, y,
                        color=cpalette[i], linestyle=linestyle, linewidth=linewidth)

            ax.fill_betweenx(y, fy1+offset, fy2+offset, alpha=alpha,
                             where=(fy1 + offset >= fy2 + offset), color=cpalette[i])

            if 'cor' not in quantity and not half:
                ax.fill_betweenx(y, -fy1+offset, -fy2+offset, alpha=alpha,
                                 where=(fy1 + offset >= fy2 + offset), color=cpalette[i])

            if not half:
                perc = x[0] * qax[quantity][1]
                idcs = [0] * 3
                percs = [0] * 3
                for pidx in range(1, len(x)):
                    if perc + x[pidx] * qax[quantity][1] <= 0.25:
                        idcs[0] = pidx
                        percs[0] += x[pidx] * qax[quantity][1]

                    if perc + x[pidx] * qax[quantity][1] <= 0.5:
                        idcs[1] = pidx
                        percs[1] += x[pidx] * qax[quantity][1]

                    if perc + x[pidx] * qax[quantity][1] <= 0.75:
                        idcs[2] = pidx
                        percs[2] += x[pidx] * qax[quantity][1]

                    perc += x[pidx] * qax[quantity][1]

                hbin = qax[quantity][1]/2
                ax.plot([offset, offset], [y[idcs[0]] + hbin, y[idcs[2]] + hbin],
                        linewidth=2, color='k')
                ax.plot([offset], [y[idcs[1]] + hbin], linewidth=3,
                        marker='o', markersize=3, color=cpalette[i])

    return ax


def plot_w_stats(data, path, args, axs=None, quantities=None, half=False, vertical=True, qtype='avg', skip=[], cpalette=None, linestyle='-'):
    global table_data, all_pdfs

    if quantities is None:
        quantities = data[list(data.keys())[0]].keys()

    with Progress() as progress:
        task1 = progress.add_task('Plotting quantities', total=len(quantities))

        for idx, q in enumerate(quantities):
            # for q in ['thetaw', 'phi', 'psi']:
            dists = {}
            for e in sorted(data.keys()):
                if e in skip:
                    continue

                task2 = progress.add_task(
                    '{} [{}]'.format(q, e), total=len(data.keys()))

                for dtype in data[e][q].keys():
                    if dtype not in dists.keys():
                        dists[dtype] = {}
                        dists[dtype]['mean'] = []
                        dists[dtype]['sd'] = []

                    means = np.array(data[e][q][dtype]['means'])
                    sds = np.array(data[e][q][dtype]['sds'])
                    if e not in table_data.keys():
                        table_data[e] = {}
                        all_pdfs[e] = {}
                    if q not in table_data[e].keys():
                        table_data[e][q] = {}
                        all_pdfs[e][q] = {}
                    if dtype not in table_data[e][q].keys():
                        table_data[e][q][dtype] = {}

                    if 'cor' not in q:
                        table_data[e][q][dtype]['means_mean'] = np.mean(means)
                        table_data[e][q][dtype]['means_sd'] = np.std(means)
                        table_data[e][q][dtype]['sds_mean'] = np.mean(sds)
                        table_data[e][q][dtype]['sds_sd'] = np.std(sds)

                    if 'cor' in q:
                        hist = np.array(data[e][q][dtype]['cor'])
                        samples = np.array(data[e][q][dtype]['num_data'])
                    else:
                        hist = np.array(data[e][q][dtype]['N'])
                        # here we always take the samples from avg
                        samples = np.array(data[e][q]['avg']['samples'])
                        if dtype != 'avg':
                            samples /= 2

                    # normalize data
                    norm_hist = np.copy(hist)
                    for m in range(hist.shape[0]):
                        if 'cor' in q:
                            norm_hist[m, :] /= samples[m, :]
                        else:
                            norm_hist[m, :] /= samples[m]
                    # now we have PDFs in the matrix
                    pdfs = norm_hist / qax[q][1]

                    d = np.mean(pdfs, axis=0)

                    all_pdfs[e][q][dtype] = d

                    # compute sd
                    sd = np.copy(hist)
                    for m in range(pdfs.shape[0]):
                        sd[m, :] = (hist[m, :] - np.mean(hist, axis=0)) ** 2
                    sd = np.sqrt(sd)
                    for m in range(pdfs.shape[0]):
                        if 'cor' in q:
                            sd[m, :] /= samples[m, :]
                        else:
                            sd[m, :] /= samples[m]
                    sd = np.mean(sd, axis=0) / qax[q][1]

                    if USE_CI:
                        dplus = np.copy(d)
                        dminus = np.copy(d)
                        for n in range(pdfs.shape[1]):

                            s = 0
                            for m in range(pdfs.shape[0]):
                                if pdfs[m, n] < d[n]:
                                    # if s > ci / 2.:
                                    #     break

                                    if s > ci / 2.:
                                        break
                                    else:
                                        s += pdfs[m, n]
                                        dminus[n] = pdfs[m, n]
                                else:
                                    break

                            s = 0
                            for m in range(pdfs.shape[0]):
                                if pdfs[m, n] > d[n]:
                                    # if s > ci / 2.:
                                    #     break

                                    if s > ci / 2.:
                                        break
                                    else:
                                        s += pdfs[m, n]
                                        dplus[n] = pdfs[m, n]
                                else:
                                    continue

                    else:
                        dplus = d + np.mean(sd, axis=0)
                        dminus = d - np.mean(sd, axis=0)

                    dists[dtype]['mean'].append(d)
                    dists[dtype]['sd'].append((dplus, dminus))

                progress.update(task2, advance=1)

            custom_vplot(dists, q, axs[idx], args,
                         half=half, vertical=vertical, qtype=qtype, cpalette=cpalette, linestyle=linestyle)
            progress.update(task1, advance=1)
    return axs


def plot(exp_files, path, args):
    data = {}

    for e in track(sorted(args.type), description='Loading files'):
        data[e] = {}

        dset = glob.glob(args.path + '/{}-*'.format(e))
        for f in dset:
            fname = f.split('/')[1]
            fname = fname.split('.txt')[0]
            keys = fname.split('-')

            if keys[1] not in data[e].keys():
                data[e][keys[1]] = {}
            if keys[2] not in data[e][keys[1]].keys():
                data[e][keys[1]][keys[2]] = {}

            data[e][keys[1]][keys[2]][keys[3]] = np.loadtxt(f)

    plot_individual_quantities(data, path, args)
    plot_collective_quantities(data, path, args)
    plot_correlation_quantities(data, path, args)

    # console.print(all_pdfs)
    console.print(table_data)
    hellinger_distances()


_SQRT2 = np.sqrt(2)


def hellinger(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2


def hellinger_distances():
    import itertools

    exps = list(all_pdfs.keys())
    num_exps = len(exps)
    c = list(itertools.combinations(exps, 2))
    pairs = set(c)

    hd = {}
    for pair in pairs:
        hd[pair] = {}

        quantities = all_pdfs[pair[0]].keys()
        for q in quantities:
            if 'cor' in q:
                continue

            hd[pair][q] = {}

            # dtypes = all_pdfs[pair[0]][q].keys()
            dtypes = ['avg']
            for dtype in dtypes:
                hd[pair][q][dtype] = hellinger(
                    all_pdfs[pair[0]][q][dtype], all_pdfs[pair[1]][q][dtype])

    console.print(hd)


def plot_individual_quantities(data, path, args):
    # individual quantities
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2])

    # -- main grid row
    grow1 = gs[1].subgridspec(
        1, 3, wspace=0.35, hspace=0.0)

    # ---- velocity
    gv = grow1[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = plot_w_stats(data, path, args, axs=[ax], quantities=['velocity'])[0]

    # ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 32])

    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.grid(False)
    ax.set_xticklabels([])
    ax.xaxis.grid(False)

    ax.set_xlabel('PDF')
    ax.set_ylabel(r'$V$ (cm/s)')

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    # ax2 = inset_axes(parent_axes, 1, 1)
    # ax2.plot([0.75, 1.05], [1.5, 30.5])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])

    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    # ax2.set_axes_locator(ip)
    ax3.set_axes_locator(ip)
    # mark_inset(parent_axes, ax2, 3, 4, ec='gray', alpha=0.3, linestyle='--')

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['velocity'], qtype='ind0', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle=':')[0]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['velocity'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    yscale = 100
    ax3 = annot_axes(ax3,
                     r'$V$ (cm/s)', r'PDF $(\times {})$'.format(yscale),
                     [0.0, 31.0], [0.0, 11],
                     [10, 2], [5, 1],
                     yscale)

    # ---- distance to the wall
    gi = grow1[0, 1].subgridspec(1, 1)
    ax = fig.add_subplot(gi[0, 0])
    ax = plot_w_stats(data, path, args, axs=[ax], quantities=['dw'])[0]

    # ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 25])

    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.grid(False)
    ax.set_xticklabels([])
    ax.xaxis.grid(False)

    ax.set_xlabel('PDF')
    ax.set_ylabel(r'$r_{\rm w}$ (cm)')

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    # ax2 = inset_axes(parent_axes, 1, 1)
    # ax2.plot([0.75, 1.05], [1.5, 30.5])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])

    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    # ax2.set_axes_locator(ip)
    ax3.set_axes_locator(ip)
    # mark_inset(parent_axes, ax2, 3, 4, ec='gray', alpha=0.3, linestyle='--')

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['dw'], qtype='ind0', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle=':')[0]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['dw'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    yscale = 100
    ax3 = annot_axes(ax3,
                     r'$r_{\rm w}$ (cm)', r'PDF $(\times {})$'.format(yscale),
                     [0.0, 25.0], [0.0, 16.5],
                     [5, 1], [5, 1],
                     yscale)

    # ---- theta
    gt = grow1[0, 2].subgridspec(1, 1)
    ax = fig.add_subplot(gt[0, 0])
    ax = plot_w_stats(data, path, args, axs=[ax], quantities=['thetaw'])[0]

    # ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 180])

    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.grid(False)
    ax.set_xticklabels([])
    ax.xaxis.grid(False)

    ax.set_xlabel('PDF')
    ax.set_ylabel(r'$\theta_{\rm w}$ $(^{\circ})$')

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    # ax2 = inset_axes(parent_axes, 1, 1)
    # ax2.plot([0.75, 1.05], [1.5, 30.5])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])

    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    # ax2.set_axes_locator(ip)
    ax3.set_axes_locator(ip)
    # mark_inset(parent_axes, ax2, 3, 4, ec='gray', alpha=0.3, linestyle='--')

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['thetaw'], qtype='ind0', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle=':')[0]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['thetaw'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim(0, 25)
    ax3.set_ylim(0, 25)
    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    yscale = 100
    ax3 = annot_axes(ax3,
                     r'$\theta_{\rm w}$ $(^{\circ})$', r'PDF $(\times {})$'.format(
                         yscale),
                     [0, 180], [0, 3.5],
                     [45, 15], [1.0, 0.25],
                     yscale)
    plt.tight_layout()
    plt.savefig(path + '/individual_quantities.tiff', dpi=600)


def plot_collective_quantities(data, path, args):
    # collective quantities
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2])

    # -- main grid row
    grow1 = gs[1].subgridspec(
        1, 3, wspace=0.35, hspace=0.0)

    # ---- d
    gv = grow1[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = plot_w_stats(data, path, args, axs=[ax], quantities=['d'])[0]

    # ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 30])

    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.grid(False)
    ax.set_xticklabels([])
    ax.xaxis.grid(False)

    ax.set_xlabel('PDF')
    ax.set_ylabel(r'$d_{ij}$ (cm)')

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    # ax2 = inset_axes(parent_axes, 1, 1)
    # ax2.plot([0.75, 1.05], [1.5, 30.5])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])

    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    # ax2.set_axes_locator(ip)
    ax3.set_axes_locator(ip)
    # mark_inset(parent_axes, ax2, 3, 4, ec='gray', alpha=0.3, linestyle='--')

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['d'], qtype='avg', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='-')[0]
    # ax3 = plot_w_stats(data, path, args, axs=[
    #     ax3], quantities=['d'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    yscale = 100
    ax3 = annot_axes(ax3,
                     r'$d_{ij}$ (cm)',
                     r'PDF $(\times {})$'.format(yscale),
                     [0.0, 30.0], [0.0, 10.0],
                     #    [0.0, 35.0], [0.0, 15.0],
                     [5, 2.5], [3, 1.5],
                     yscale)

    # ---- phi
    gv = grow1[0, 1].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = plot_w_stats(data, path, args, axs=[ax], quantities=['phi'])[0]

    # ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 180])

    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.grid(False)
    ax.set_xticklabels([])
    ax.xaxis.grid(False)

    ax.set_xlabel('PDF')
    ax.set_ylabel(r'$\phi_{ij}$ $(^{\circ})$')

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    # ax2 = inset_axes(parent_axes, 1, 1)
    # ax2.plot([0.75, 1.05], [1.5, 30.5])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])

    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    # ax2.set_axes_locator(ip)
    ax3.set_axes_locator(ip)
    # mark_inset(parent_axes, ax2, 3, 4, ec='gray', alpha=0.3, linestyle='--')

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['phi'], qtype='avg', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='-')[0]
    # ax3 = plot_w_stats(data, path, args, axs=[
    #     ax3], quantities=['d'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    yscale = 100
    ax3 = annot_axes(ax3,
                     r'$\phi_{ij}$ $(^{\circ})$', r'PDF $(\times {})$'.format(
                         yscale),
                     [0, 180], [0, 1.5],
                     [45, 15], [1.0, 0.25],
                     yscale)

    # ---- psi
    gv = grow1[0, 2].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = plot_w_stats(data, path, args, axs=[ax], quantities=['psi'])[0]

    # ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([-180, 180])

    ax.yaxis.set_major_locator(MultipleLocator(45))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    # ax.yaxis.grid(False)
    ax.set_xticklabels([])
    ax.xaxis.grid(False)

    ax.set_xlabel('PDF')
    ax.set_ylabel(r'$\psi_{ij}$ $(^{\circ})$')

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    # ax2 = inset_axes(parent_axes, 1, 1)
    # ax2.plot([0.75, 1.05], [1.5, 30.5])
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])

    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    # ax2.set_axes_locator(ip)
    ax3.set_axes_locator(ip)
    # mark_inset(parent_axes, ax2, 3, 4, ec='gray', alpha=0.3, linestyle='--')

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['psi'], qtype='ind0', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle=':')[0]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['psi'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    yscale = 100
    ax3 = annot_axes(ax3,
                     r'$\psi_{ij}$ $(^{\circ})$', r'PDF $(\times {})$'.format(
                         yscale),
                     [-180, 180], [0, 0.75],
                     [60, 15], [0.25, 0.05],
                     yscale)
    plt.tight_layout()
    plt.savefig(path + '/collective_quantities.tiff', dpi=600)


def plot_correlation_quantities(data, path, args):
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(5)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 2])

    # -- main grid row
    grow1 = gs[1].subgridspec(
        1, 3, wspace=0.45, hspace=0.0)

    # ---- corx
    gv = grow1[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = plot_w_stats(data, path, args, axs=[
        ax], quantities=['corx'], qtype='avg', half=True, vertical=False, cpalette=palette, linestyle='-')[0]

    ax = annot_axes(ax,
                    '$t$ (s)', r'$C_X$ $(cm^2)$',
                    [0.0, 30.0], [0.0, 1450],
                    [5, 2.5], [250, 125],
                    1)

    ax.yaxis.set_major_locator(MultipleLocator(200))
    ax.yaxis.set_minor_locator(MultipleLocator(50))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])
    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    ax3.set_axes_locator(ip)

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['corx'], qtype='ind0', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle=':')[0]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['corx'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    ax3 = annot_axes(ax3,
                     '$t$ (s)', r'$C_X$ $(cm^2)$',
                     [0.0, 30.0], [0.0, 1100],
                     [5, 2.5], [250, 125],
                     1)

    # ---- corv
    gv = grow1[0, 1].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = plot_w_stats(data, path, args, axs=[
        ax], quantities=['corv'], qtype='avg', half=True, vertical=False, cpalette=palette, linestyle='-')[0]

    ax = annot_axes(ax,
                    '$t$ (s)', r'$C_V$ $(\,cm^2 / \,s^2)$',
                    [0.0, 30.0], [-100.0, 200],
                    [5, 2.5], [50, 25],
                    1)

    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])
    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    ax3.set_axes_locator(ip)

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['corv'], qtype='ind0', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle=':')[0]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['corv'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    ax3 = annot_axes(ax3,
                     '$t$ (s)', r'$C_V$ $(\,cm^2 / \,s^2)$',
                     [0.0, 30.0], [-50.0, 200],
                     [5, 2.5], [50, 25],
                     1)

    # ---- cortheta
    gv = grow1[0, 2].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = plot_w_stats(data, path, args, axs=[
        ax], quantities=['cortheta'], qtype='avg', half=True, vertical=False, cpalette=palette, linestyle='-')[0]

    ax = annot_axes(ax,
                    '$t$ (s)', r'$C_{\theta_{\rm w}}$',
                    [0.0, 30.0], [0.0, 1.0],
                    [5, 2.5], [0.2, 0.1],
                    1)

    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    # inset
    parent_axes = plt.gca()
    ax3 = plt.gcf().add_axes([0, 0.5, 1, 1])
    ip = InsetPosition(parent_axes, [0.0, 1.22, 1.0, 0.4])
    ax3.set_axes_locator(ip)

    cpalette = [palette[-1]]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['cortheta'], qtype='ind0', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle=':')[0]
    ax3 = plot_w_stats(data, path, args, axs=[
        ax3], quantities=['cortheta'], qtype='ind1', half=True, vertical=False, skip=SKIP_FOLDERS, cpalette=cpalette, linestyle='--')[0]

    ax3.set_xlim([0.75, 1.05])
    for d in ["left", "right", "top", "bottom"]:
        ax3.spines[d].set_linewidth(0.5)

    ax3 = annot_axes(ax3,
                     '$t$ (s)', r'$C_{\theta_{\rm w}}$',
                     [0.0, 30.0], [-0.0, 1.0],
                     [5, 2.5], [0.2, 0.1],
                     1)
    plt.tight_layout()
    plt.savefig(path + '/correlation_quantities.tiff', dpi=600)
