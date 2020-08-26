#!/usr/bin/env python
import argparse
import glob

import matplotlib.lines as mlines
import seaborn as sns
from pylab import *
from itertools import cycle

from plot_geometrical_leader_info import compute_leadership
from utils.features import Velocities

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
# palette = sns.cubehelix_palette(11)
palette = sns.color_palette("Set1", n_colors=11, desat=.5)
colors = sns.color_palette(palette)
colorcycler = cycle(colors)

sns.set(style="darkgrid")

lines = ["-","--","-.",":"]
linecycler = cycle(lines)

gfontsize = 10
params = {
    'axes.labelsize': gfontsize,
    'font.size': gfontsize,
    'legend.fontsize': gfontsize,
    'xtick.labelsize': gfontsize,
    'ytick.labelsize': gfontsize,
    'text.usetex': False,
    # 'figure.figsize': [10, 15]
    # 'ytick.major.pad': 4,
    # 'xtick.major.pad': 4,
    'font.family': 'Arial',
}
rcParams.update(params)

pts = np.linspace(0, np.pi * 2, 24)
circ = np.c_[np.sin(pts) / 2, -np.cos(pts) / 2]
vert = np.r_[circ, circ[::-1] * 1.0]
open_circle = mpl.path.Path(vert)

extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
                  edgecolor='none', linewidth=0)

shapeList = []


def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif

def relative_orientation_to_neigh(data, experiments):
    num_experiments = len(data.keys())
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    labels = []

    for _, k in enumerate(sorted(data.keys())):
        labels.append(k)

        num_exp = len(data[k]['pos'])
        for e in range(num_exp):
            p = data[k]['pos'][e] 
            v = data[k]['vel'][e] 

            hdgs = np.empty((p.shape[0], 0))
            for i in range(p.shape[1] // 2):
                hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                hdg = np.array(list(map(angle_to_pipi, hdg)))
                hdgs = np.hstack((hdgs, hdg.reshape(-1, 1)))
            
        assert (p.shape[1] // 2) == 2
        angle_dif = hdgs[:, 0] - hdgs[:, 1]        
        angle_dif = np.array(list(map(angle_to_pipi, angle_dif))) * 180 / np.pi
        
        sns.kdeplot(angle_dif, ax=ax, color=next(colorcycler), linestyle=next(linecycler), label='Leader (' + k + ')')

    ax.set_xlabel('Relative angle to neighbour in degrees')
    ax.set_ylabel('KDE')
    plt.savefig('relative_orientation_neigh.png', dpi=300)


def relative_orientation_to_wall(data, experiments):
    num_experiments = len(data.keys())
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    labels = []

    leadership = {}
    for i, k in enumerate(sorted(data.keys())):
        pos = data[k]['pos']
        vel = data[k]['vel']

        leadership[k] = []
        for idx in range(len(pos)):
            (_, leadership_timeseries) = compute_leadership(pos[idx], vel[idx])

            # TODO: don't do it globally but locally and then reset the focal window
            # this is to sanitize the leadership timeseries
            window = 4
            hwindow = window // 2
            lt = np.array(leadership_timeseries)
            for l in range(0, lt.shape[0], hwindow):
                lb = max([0, l - hwindow])
                ub = min([l + hwindow, lt.shape[0]]) 

                snap = list(lt[lb:ub, 1])
                fel = max(set(snap), key = snap.count) 
                lt[lb:ub, 1] = fel
                leadership_timeseries[lb:ub] = lt[lb:ub].tolist()

            leadership[k].append(leadership_timeseries)

    for _, k in enumerate(sorted(data.keys())):
        labels.append(k)

        leader_angles = []
        follower_angles = []
        num_exp = len(data[k]['pos'])
        for e in range(num_exp):
            p = data[k]['pos'][e] 
            v = data[k]['vel'][e] 
            l = np.array(leadership[k][e])

            rel_angles = np.empty((p.shape[0], 0))
            for i in range(p.shape[1] // 2):
                theta = np.arctan2(p[:, i*2+1], p[:, i*2])
                theta = np.array(list(map(angle_to_pipi, theta)))

                hdg = np.arctan2(v[:, i*2+1], v[:, i*2])
                hdg = np.array(list(map(angle_to_pipi, hdg)))

                rel_angle = theta - hdg
                rel_angle = np.array(list(map(angle_to_pipi, rel_angle))) * 180 / np.pi
                rel_angle = np.abs(rel_angle)
                rel_angles = np.hstack((rel_angles, rel_angle.reshape(-1, 1)))

            num_individuals = p.shape[1] // 2
            for j in range(num_individuals):
                idx_leaders = np.where(l[:, 1] == j)
                leader_angles += rel_angles[idx_leaders, j].tolist()[0]
                follower_idcs = list(range(num_individuals))
                follower_idcs.remove(j)
                for fidx in follower_idcs:
                    follower_angles += rel_angles[idx_leaders, fidx].tolist()[0]

        sns.kdeplot(leader_angles, ax=ax, color=next(colorcycler), linestyle=next(linecycler), label='Leader (' + k + ')')
        sns.kdeplot(follower_angles, ax=ax, color=next(colorcycler), linestyle=next(linecycler), label='Follower (' + k + ')')

    ax.set_xlabel('Relative angle to wall in degrees')
    ax.set_ylabel('KDE')
    plt.savefig('relative_orientation.png', dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resultant velocity histogram figure')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to data directory',
                        required=True)
    parser.add_argument('--radius', '-r', type=float,
                        help='Raidus',
                        default=0.25,
                        required=False)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    args = parser.parse_args()

    experiments = {
        'Hybrid': '*generated_positions.dat',
        'Virtual': '*generated_virtu_positions.dat',
        'Real': '*processed_positions.dat',
    }

    palette = sns.cubehelix_palette(len(experiments.keys()))
    colors = sns.color_palette(palette)

    for i in range(len(experiments.keys())):
        shapeList.append(Circle((0, 0), radius=1, facecolor=colors[i]))

    data = {}
    for e in sorted(experiments.keys()):
        pos = glob.glob(args.path + '/' + experiments[e])
        if len(pos) == 0:
            continue
        data[e] = {'pos': [], 'vel': []}
        for p in pos:
            p = np.loadtxt(p) * args.radius
            v = Velocities([p], args.timestep).get()[0]
            data[e]['pos'].append(p)
            data[e]['vel'].append(v)

    relative_orientation_to_wall(data, experiments)
    relative_orientation_to_neigh(data, experiments)
