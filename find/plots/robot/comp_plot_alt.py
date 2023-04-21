#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities, Accelerations
from find.utils.utils import angle_to_pipi, compute_leadership
from find.plots.common import *

from find.plots.robot.velocity_bplots import vel_plots
from find.plots.robot.distance_to_wall_bplots import dtw_plots
from find.plots.robot.relative_orientation_bplots import relor_wall_plots, relor_neigh_plots, viewing_plots
from find.plots.robot.acceleration_bplots import acc_plots
from find.plots.robot.interindividual_dist_bplots import idist_plots, cdist_plots
from find.plots.robot.occupancy_grids import grid_plot, grid_plot_singles
from find.plots.robot.activity_bplots import activity_plots
import find.plots.spatial.grid_occupancy as go

import find.plots.spatial.resultant_velocity as rv
import find.plots.spatial.distance_to_wall as dtw
import find.plots.spatial.relative_orientation as relor
import find.plots.spatial.interindividual_distance as interd
# from find.plots.dl_si_2021.individual_quantities import annot_axes

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)


def stars(p):
    if p < 0.0001:
        return "****"
    elif (p < 0.001):
        return "***"
    elif (p < 0.01):
        return "**"
    elif (p < 0.05):
        return "*"
    else:
        return "-"


def stat(ax, idcs, y_start, annot_pos, p_value=-1, scale=1.02):
    if p_value >= 0:
        offset = 0.02
        ax.annotate("", xy=(idcs[0] + offset, y_start + 1.5), xycoords='data',
                    xytext=(idcs[1] - offset, y_start + 1.5), textcoords='data',
                    arrowprops=dict(arrowstyle="-", ec='#aaaaaa',
                                    connectionstyle="bar,fraction=0.22"))
        ax.text((idcs[0] + idcs[1])/2, y_start + 0.01 + annot_pos, stars(p_value),
                horizontalalignment='center',
                verticalalignment='center')


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


def individual_quantities_bplots_pair(data, grids, path, args):
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot'].copy()

    palette = ["#807f7d", "#3498db", "#e74c3c"]

    fig = plt.figure()

    fig.set_figwidth(10)
    fig.set_figheight(6)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    # -- main grid row
    # grow0 = gs[0].subgridspec(
    #     1, 1, wspace=0.0, hspace=0.0)
    grow1 = gs[0].subgridspec(
        1, 3, wspace=0.25, hspace=0.0)

    # ---- velocity
    gv = grow1[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(gv[0, 0])
    ax = vel_plots(data, path, ax, args, orient='v',
                   width=0.6, palette=palette)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 70])

    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(False)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    yticks = ax.yaxis.get_major_ticks()
    for i in range(6, len(yticks)):
        yticks[i].set_visible(False)

    yticks = ax.yaxis.get_minor_ticks()
    for i in range(37, len(yticks)):
        yticks[i].set_visible(False)

    # inset
    axin = ax.inset_axes([0.5, 1.25, 0.5, 0.5])
    set_uni_palette(["#e74c3c"])
    axin = rv.compute_resultant_velocity(sub_data, axin, args, [0, 41])
    # axin.set_ylim(1.8, 2.2)
    axin.set_ylim(0, 43)
    axin.set_xlim([1.65, 2.35])
    ax.indicate_inset_zoom(axin, alpha=0.2)
    for d in ["left", "right", "top", "bottom"]:
        axin.spines[d].set_linewidth(0.5)

    yscale = 100
    axin = annot_axes(axin,
                      r'$V$ (cm/s)', r'PDF $(\times {})$'.format(yscale),
                      [0.0, 31.0], [0.0, 11],
                      [10, 2], [5, 1],
                      yscale)

    # stats
    pvalues = [0.83, 0.012, 0.06]
    stat(ax, (0, 1), 41,  4.5, p_value=pvalues[0])
    stat(ax, (1, 2), 41,  3, p_value=pvalues[1])
    stat(ax, (0, 2), 49,  12, p_value=pvalues[2])

    # ---- distance to wall
    gd = grow1[0, 1].subgridspec(1, 1, wspace=0.0, hspace=0.5)
    ax = fig.add_subplot(gd[0, 0])

    ax = dtw_plots(data, path, ax, args, orient='v',
                   width=0.6, palette=palette)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 43.5])

    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(False)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    yticks = ax.yaxis.get_major_ticks()
    for i in range(7, len(yticks)):
        yticks[i].set_visible(False)

    yticks = ax.yaxis.get_minor_ticks()
    for i in range(21, len(yticks)):
        yticks[i].set_visible(False)

    # inset
    axin = ax.inset_axes([0.5, 1.25, 0.5, 0.5])
    set_uni_palette(["#e74c3c"])
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot'].copy()
    axin = dtw.distance_plot(sub_data, axin, args, [0, 25])
    # axin.set_ylim(1.8, 2.2)
    axin.set_ylim(0, 26)
    axin.set_xlim([1.7, 2.3])
    ax.indicate_inset_zoom(axin, alpha=0.2)
    for d in ["left", "right", "top", "bottom"]:
        axin.spines[d].set_linewidth(0.5)

    yscale = 100
    axin = annot_axes(axin,
                      r'$r_w$ (cm)', r'PDF $(\times {})$'.format(yscale),
                      [0.0, 25.0], [0.0, 15],
                      [5, 1], [5, 1],
                      yscale)

    # stats
    pvalues = [0.0223, 0.99, 0.02]
    stat(ax, (0, 1), 25,  2.5, p_value=pvalues[0])
    stat(ax, (1, 2), 25,  3, p_value=pvalues[1])
    stat(ax, (0, 2), 30,  7, p_value=pvalues[2])

    # ---- relor
    go = grow1[0, 2].subgridspec(1, 1, wspace=0.0, hspace=0.5)
    ax = fig.add_subplot(go[0, 0])

    ax = relor_wall_plots(data, path, ax, args, orient='v', palette=palette)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 310])

    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(False)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    yticks = ax.yaxis.get_major_ticks()
    for i in range(8, len(yticks)):
        yticks[i].set_visible(False)

    yticks = ax.yaxis.get_minor_ticks()
    for i in range(13, len(yticks)):
        yticks[i].set_visible(False)

    # inset
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot']
    axin = ax.inset_axes([0.5, 1.25, 0.5, 0.5])
    set_uni_palette(["#e74c3c"])
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot'].copy()
    axin = relor.relative_orientation_to_wall(sub_data, axin, args)
    # axin.set_ylim(1.8, 2.2)
    axin.set_ylim(0, 185)
    axin.set_xlim([1.7, 2.3])
    ax.indicate_inset_zoom(axin, alpha=0.2)
    for d in ["left", "right", "top", "bottom"]:
        axin.spines[d].set_linewidth(0.5)

    yscale = 100
    axin = annot_axes(axin,
                      r'$\theta_{\rm w}$ $(^{\circ})$', r'PDF $(\times {})$'.format(
                          yscale),
                      [0, 180], [0, 2.5],
                      [45, 15], [1.0, 0.25],
                      yscale)

    # stats
    pvalues = [0.0, 0.53, 0.00]
    stat(ax, (0, 1), 190,  10, p_value=pvalues[0])
    stat(ax, (1, 2), 190,  12, p_value=pvalues[1])
    stat(ax, (0, 2), 225,  43, p_value=pvalues[2])

    plt.tight_layout()
    plt.savefig(path + 'individual_quantities_bplots.tiff',
                bbox_inches='tight', dpi=600)


def collective_quantities_bplots_pair(data, grids, path, args):
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot'].copy()

    palette = ["#807f7d", "#3498db", "#e74c3c"]

    fig = plt.figure()

    fig.set_figwidth(10)
    fig.set_figheight(6)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

    # -- main grid row
    # grow0 = gs[0].subgridspec(
    #     1, 1, wspace=0.0, hspace=0.0)
    grow1 = gs[0].subgridspec(
        1, 3, wspace=0.25, hspace=0.0)

    # ---- interindividual
    gi = grow1[0, 0].subgridspec(1, 1)
    ax = fig.add_subplot(gi[0, 0])
    ax = idist_plots(data, path, ax, args, orient='v',
                     width=0.6, palette=palette)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 84])

    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(False)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    yticks = ax.yaxis.get_major_ticks()
    for i in range(7, len(yticks)):
        yticks[i].set_visible(False)

    yticks = ax.yaxis.get_minor_ticks()
    for i in range(46, len(yticks)):
        yticks[i].set_visible(False)

    # inset
    axin = ax.inset_axes([0.5, 1.25, 0.5, 0.5])
    set_uni_palette(["#e74c3c"])
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot'].copy()
    sd = {}
    sd['3_Robot'] = sub_data['3_Robot']['idist']
    axin = interd.interindividual_distance(sd, axin, args, [0, 50])
    # axin.set_ylim(1.8, 2.2)
    axin.set_ylim(0, 38)
    # axin.set_xlim([1.7, 2.3])
    # ax.indicate_inset_zoom(axin, alpha=0.2)
    # for d in ["left", "right", "top", "bottom"]:
    #     axin.spines[d].set_linewidth(0.5)

    yscale = 100
    axin = annot_axes(axin,
                      r'$d_{ij}$ (cm)',
                      r'PDF $(\times {})$'.format(yscale),
                      [0.0, 25.0], [0.0, 10.0],
                      #    [0.0, 35.0], [0.0, 15.0],
                      [5, 2.5], [3, 1.5],
                      yscale)

    # stats
    pvalues = [0.012, 0.0286, 0.83]
    stat(ax, (0, 1), 50,  4, p_value=pvalues[0])
    stat(ax, (1, 2), 50,  4, p_value=pvalues[1])
    stat(ax, (0, 2), 60,  13, p_value=pvalues[2])

    # ---- phi
    gd = grow1[0, 1].subgridspec(1, 1, wspace=0.0, hspace=0.5)
    ax = fig.add_subplot(gd[0, 0])

    ax = relor_neigh_plots(data, path, ax, args, orient='v',
                           width=0.6, palette=palette)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([0, 310])

    ax.yaxis.set_major_locator(MultipleLocator(30))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(False)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)

    yticks = ax.yaxis.get_major_ticks()
    for i in range(8, len(yticks)):
        yticks[i].set_visible(False)

    yticks = ax.yaxis.get_minor_ticks()
    for i in range(13, len(yticks)):
        yticks[i].set_visible(False)

    # inset
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot']
    axin = ax.inset_axes([0.5, 1.25, 0.5, 0.5])
    set_uni_palette(["#e74c3c"])
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot'].copy()
    axin = relor.relative_orientation_to_neigh(sub_data, axin, args)
    # axin.set_ylim(1.8, 2.2)
    axin.set_ylim(0, 185)
    # axin.set_xlim([1.7, 2.3])
    # ax.indicate_inset_zoom(axin, alpha=0.2)
    # for d in ["left", "right", "top", "bottom"]:
    #     axin.spines[d].set_linewidth(0.5)

    yscale = 100
    axin = annot_axes(axin,
                      r'$\phi_{ij}$ $(^{\circ})$', r'PDF $(\times {})$'.format(
                          yscale),
                      [0, 180], [0, 2.5],
                      [45, 15], [1.0, 0.25],
                      yscale)

    # stats
    pvalues = [0.005, 0.0584, 0]
    stat(ax, (0, 1), 190,  12, p_value=pvalues[0])
    stat(ax, (1, 2), 190,  15, p_value=pvalues[1])
    stat(ax, (0, 2), 225,  42, p_value=pvalues[2])

    # ---- psi
    go = grow1[0, 2].subgridspec(1, 1, wspace=0.0, hspace=0.5)
    ax = fig.add_subplot(go[0, 0])

    ax = viewing_plots(data, path, ax, args, orient='v', palette=palette)
    ax.set_xlim([-0.5, 2.5])
    ax.set_ylim([-180, 405])

    ax.yaxis.set_major_locator(MultipleLocator(45))
    ax.yaxis.set_minor_locator(MultipleLocator(15))
    ax.tick_params(axis='y', which='both', bottom=True,
                   left=True, right=True, top=True)
    ax.tick_params(axis="y", which='both', direction="in")
    ax.set_xticklabels([])
    ax.set_title('')
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.yaxis.grid(False)

    ax.tick_params(axis="x", which='major', length=3, width=0.7)
    ax.tick_params(axis="x", which='minor', length=1.5, width=0.7)
    ax.tick_params(axis="y", which='major', length=3, width=0.7)
    ax.tick_params(axis="y", which='minor', length=1.5, width=0.7)
    ax.yaxis.set_label_coords(-.13, .5)

    yticks = ax.yaxis.get_major_ticks()
    for i in range(10, len(yticks)):
        yticks[i].set_visible(False)

    yticks = ax.yaxis.get_minor_ticks()
    for i in range(17, len(yticks)):
        yticks[i].set_visible(False)

    # inset
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot']
    axin = ax.inset_axes([0.5, 1.25, 0.5, 0.5])
    set_uni_palette(["#e74c3c"])
    sub_data = {}
    sub_data['3_Robot'] = data['3_Robot'].copy()
    axin = relor.viewing_angle(sub_data, axin, args)
    # axin.set_ylim(1.8, 2.2)
    axin.set_ylim(-180, 185)
    # axin.set_xlim([1.7, 2.3])
    # ax.indicate_inset_zoom(axin, alpha=0.2)
    # for d in ["left", "right", "top", "bottom"]:
    #     axin.spines[d].set_linewidth(0.5)

    yscale = 100
    axin = annot_axes(axin,
                      r'$\psi_{ij}$ $(^{\circ})$', r'PDF $(\times {})$'.format(
                          yscale),
                      [-180, 180], [0, 2.5],
                      [45, 15], [1.0, 0.25],
                      yscale)

    # stats
    pvalues = [0.45, 0.97, 0.31]
    stat(ax, (0, 1), 190,  26, p_value=pvalues[0])
    stat(ax, (1, 2), 190,  25, p_value=pvalues[1])
    stat(ax, (0, 2), 255,  80, p_value=pvalues[2])

    plt.tight_layout()
    plt.savefig(path + 'collective_quantities_bplots.tiff',
                bbox_inches='tight', dpi=600)


def plot(exp_files, path, args):
    data = {}
    grids = {}
    num_inds = -1

    for e in sorted(exp_files.keys()):
        samples = 0

        if e == 'BOBI' or 'Simu' in e:
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
        data[e]['racc'] = []
        data[e]['idist'] = []
        data[e]['cdist'] = []
        data[e]['distance_to_wall'] = []
        data[e]['theta'] = []
        data[e]['phi'] = []
        data[e]['psi'] = []
        if 'Simu' not in e:
            data[e]['samples'] = np.loadtxt(
                args.path + '/' + e + '/sample_counts.txt')
        else:
            data[e]['samples'] = np.array([0, 0])
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
            accelerations = Accelerations(
                [velocities], timestep).get()[0]

            samples += positions.shape[0]
            num_inds = positions.shape[1] // 2

            if args.robot:
                r = p.replace('.dat', '_ridx.dat')
                ridx = np.loadtxt(r).astype(int)
                data[e]['ridx'].append(int(ridx))

            tup = []
            for i in range(velocities.shape[1] // 2):
                linear_velocity = np.sqrt(
                    velocities[:, i * 2] ** 2 + velocities[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_velocity)
            mat = np.array(tup).T
            mat = mat[np.all(mat <= 42, axis=1), :]
            data[e]['rvel'].append(mat)

            tup = []
            for i in range(accelerations.shape[1] // 2):
                linear_acceleration = np.sqrt(
                    accelerations[:, i * 2] ** 2 + accelerations[:, i * 2 + 1] ** 2).tolist()
                tup.append(linear_acceleration)
            mat = np.array(tup).T
            mat = mat[np.all(mat <= 175, axis=1), :]
            data[e]['racc'].append(mat)

            interind_dist = []
            if num_inds >= 2:
                dist = np.zeros((1, positions.shape[0]))
                for i in range(1, num_inds):
                    dist += (positions[:, 0] - positions[:, i*2]) ** 2 + \
                        (positions[:, 1] - positions[:, i*2 + 1]) ** 2
                interind_dist = np.sqrt(dist / (num_inds - 1))
                interind_dist = interind_dist.tolist()[0]
            else:
                print('Single fish. Skipping interindividual distance')

            data[e]['idist'].append(interind_dist)
            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)

            dist_mat = []
            for i in range(positions.shape[1] // 2):
                distance = args.radius - \
                    np.sqrt(positions[:, i * 2] ** 2 +
                            positions[:, i * 2 + 1] ** 2)
                dist_mat.append(distance)
            dist_mat = np.array(dist_mat).T
            data[e]['distance_to_wall'].append(dist_mat)

            # theta
            hdgs = np.empty((velocities.shape[0], 0))
            for i in range(positions.shape[1] // 2):
                hdg = np.arctan2(velocities[:, i*2+1], velocities[:, i*2])
                hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))

            fidx = 0
            angle_difs = []

            # for the focal
            angle_dif_focal = hdgs[:, fidx] - \
                np.arctan2(positions[:, fidx*2+1], positions[:, fidx*2])
            angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
            angle_dif_focal = np.array(list(
                map(lambda x: x * 180 / np.pi, angle_dif_focal)))
            angle_difs.append(np.abs(angle_dif_focal))

            # for the neigh
            for nidx in range(num_inds):
                if fidx == nidx:
                    continue
                angle_dif_neigh = hdgs[:, nidx] - \
                    np.arctan2(positions[:, nidx*2+1], positions[:, nidx*2])
                angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
                angle_dif_neigh = np.array(list(
                    map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
                angle_difs.append(np.abs(angle_dif_neigh))

            theta = []
            for i in range(len(angle_difs)):
                theta += angle_difs[i].tolist()
            data[e]['theta'].append(theta)

            # phi
            # for the focal
            angle_dif_focal = hdgs[:, 0] - hdgs[:, 1]
            angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
            angle_dif_focal = np.array(list(
                map(lambda x: x * 180 / np.pi, angle_dif_focal)))

            # for the neigh
            angle_dif_neigh = hdgs[:, 1] - hdgs[:, 0]
            angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
            angle_dif_neigh = np.array(list(
                map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
            angle_difs = [angle_dif_focal, angle_dif_neigh]

            phi = []
            for i in range(len(angle_difs)):
                phi += angle_difs[i].tolist()
            data[e]['phi'].append(phi)

            # psi
            angle_difs = []

            # for the focal
            angle_dif_focal = hdgs[:, 0] - \
                np.arctan2(positions[:, 3] - positions[:, 1],
                           positions[:, 2] - positions[:, 0])
            angle_dif_focal = list(map(angle_to_pipi, angle_dif_focal))
            angle_dif_focal = np.array(list(
                map(lambda x: x * 180 / np.pi, angle_dif_focal)))

            # for the neigh
            angle_dif_neigh = hdgs[:, 1] - \
                np.arctan2(positions[:, 1] - positions[:, 3],
                           positions[:, 0] - positions[:, 2])
            angle_dif_neigh = list(map(angle_to_pipi, angle_dif_neigh))
            angle_dif_neigh = np.array(list(
                map(lambda x: x * 180 / np.pi, angle_dif_neigh)))
            angle_difs = [angle_dif_focal, angle_dif_neigh]

            psi = []
            for i in range(len(angle_difs)):
                psi += angle_difs[i].tolist()
            data[e]['psi'].append(psi)

        print('{} has {} samples'.format(e, samples))

        # # construct grids
        # ridcs = {}
        # ridcs[e] = data[e]['ridx']
        # sdata = {}
        # sdata[e] = data[e]['pos']

        # if args.separate:
        #     x, y, z = go.construct_grid_sep(
        #         sdata, e, args, sigma=1, ridcs=ridcs)
        #     grid = {'x': x, 'y': y, 'z': z}
        #     grids[e] = grid
        # else:
        #     x, y, z = go.construct_grid(
        #         sdata, e, args, sigma=1)
        #     grid = {'x': x, 'y': y, 'z': z}
        #     grids[e] = grid

    individual_quantities_bplots_pair(data, grids, path, args)
    collective_quantities_bplots_pair(data, grids, path, args)
