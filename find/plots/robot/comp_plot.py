#!/usr/bin/env python
import glob
import argparse

from find.utils.features import Velocities, Accelerations
from find.utils.utils import compute_leadership
from find.plots.common import *

from find.plots.robot.velocity_bplots import vel_plots
from find.plots.robot.acceleration_bplots import acc_plots
from find.plots.robot.interindividual_dist_bplots import idist_plots
from find.plots.robot.occupancy_grids import grid_plot

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def comp_plot(data, path, args):
    fig = plt.figure()
    fig.set_figwidth(10)
    fig.set_figheight(9)

    gs = fig.add_gridspec(2, 1)

    ## -- circle
    grow = gs[0].subgridspec(1, 5, wspace=0.0, hspace=0.0)

    # velocity
    # gv = grow[0, 1].subgridspec(1, 2, wspace=0.0, hspace=0.0)
    # ax = [
    #     fig.add_subplot(gv[0, 0]),
    #     fig.add_subplot(gv[0, 1])
    # ]
    # ax = vel_plots(data, path, ax, args)

    # acceleration
    ga = grow[0, 2].subgridspec(1, 2, wspace=0.0, hspace=0.0)
    ax = [
        fig.add_subplot(ga[0, 0]),
        fig.add_subplot(ga[0, 1])
    ]
    ax = acc_plots(data, path, ax, args)

    # interindividual
    gi = grow[0, 3].subgridspec(1, 1, wspace=0.0, hspace=0.0)
    ax = fig.add_subplot(gi[0, 0])
    ax = idist_plots(data, path, ax, args)

    # plt.tight_layout()
    plt.savefig(path + 'comp_plot.png', bbox_inches='tight')


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
        data[e]['racc'] = []
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
            velocities = Velocities([positions], timestep).get()[0][1:-1]
            accelerations = Accelerations(
                [velocities], timestep).get()[0][1:-1]

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

            interind_dist = np.sqrt(
                (positions[:, 0] - positions[:, 2]) ** 2 + (positions[:, 1] - positions[:, 3]) ** 2).tolist()
            data[e]['idist'].append(interind_dist)

            data[e]['pos'].append(positions)
            data[e]['vel'].append(velocities)
        print('{} has {} samples'.format(e, samples))

    comp_plot(data, path, args)
