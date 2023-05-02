#!/usr/bin/env python
import re
import glob
import random
import pickle
import argparse
from tqdm import tqdm
from scipy.stats import t

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

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, FuncFormatter)


quantities_for_eval = [
    'rvel',
    'distance_to_wall',
    'theta',
    'idist',
    'phi',
    'psi'
]

quantity_bins = {
    'rvel': range(0, 84 + 1, 1),
    'distance_to_wall': range(0, 50 + 1, 1),
    'theta': range(0, 180 + 1, 1),
    'idist': range(0, 100 + 1, 1),
    'phi': range(0, 180 + 1, 1),
    'psi': range(-180, 180 + 1, 1)
}


def comp_conf(distribution, mu, ste, confidence_level):
    degrees_of_freedom = len(distribution) - 1
    t_score = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
    me = t_score * ste
    lower_bound = mu - me
    upper_bound = mu + me
    return lower_bound, upper_bound


def measure_quantity(q, data, idcs, args):
    distribution = []
    r_distribution = []
    n_distribution = []
    for idx in idcs:
        if type(data[q][idx]) is list:
            distribution += data[q][idx]
        elif type(data[q][idx]) is np.ndarray:
            if args.robot and 'ridx' in data.keys():
                ridx = data['ridx'][idx]
                if ridx < 0:
                    ridx = 0  # if there is no robot we just take fish 0 and separate it from fish 1
            else:
                ridx = 0

            # ! this is very specific, perhaps ok for this
            num_inds = data[q][idx].shape[1]
            for nidx in range(num_inds):
                distribution += data[q][idx][:, nidx].tolist()
                if ridx == nidx:
                    r_distribution += data[q][idx][:, nidx].tolist()
                else:
                    n_distribution += data[q][idx][:, nidx].tolist()

    # print('Done with dists')

    # average
    mu = np.mean(distribution)
    std = np.std(distribution)
    ste = std / np.sqrt(len(distribution))
    # h, bin_edges, _ = plt.hist(distribution, bins=quantity_bins[q])
    h, bin_edges = np.histogram(distribution, bins=quantity_bins[q])

    bin_stds = []
    d = np.array(distribution)
    for i in range(len(bin_edges) - 1):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        bin_elements = d[(d >= lower_bound) & (
            d < upper_bound)]
        bin_stds.append(np.std(bin_elements))

    bin_width = bin_edges[1] - bin_edges[0]
    # total_area = np.sum(h * bin_width)
    # pdf = h / total_area
    # lb, ub = comp_conf(pdf, bin_edges, 0.95)
    lb, ub = comp_conf(distribution, mu, ste, 0.95)

    # print('Done with avg stats')

    # robot
    r_mu = np.mean(r_distribution)
    r_std = np.std(r_distribution)
    r_ste = r_std / np.sqrt(len(r_distribution))
    r_h, bin_edges = np.histogram(r_distribution, bins=quantity_bins[q])
    bin_width = bin_edges[1] - bin_edges[0]
    r_lb, r_ub = comp_conf(r_distribution, r_mu, r_ste, 0.95)

    r_bin_stds = []
    d = np.array(r_distribution)
    for i in range(len(bin_edges) - 1):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        bin_elements = d[(d >= lower_bound) & (
            d < upper_bound)]
        r_bin_stds.append(np.std(bin_elements))

    # print('Done with ind0 stats')

    # neigh
    n_mu = np.mean(n_distribution)
    n_std = np.std(n_distribution)
    n_ste = n_std / np.sqrt(len(n_distribution))
    n_h, bin_edges = np.histogram(n_distribution, bins=quantity_bins[q])
    bin_width = bin_edges[1] - bin_edges[0]
    n_lb, n_ub = comp_conf(n_distribution, n_mu, n_ste, 0.95)

    n_bin_stds = []
    d = np.array(n_distribution)
    for i in range(len(bin_edges) - 1):
        lower_bound = bin_edges[i]
        upper_bound = bin_edges[i + 1]
        bin_elements = d[(d >= lower_bound) & (
            d < upper_bound)]
        n_bin_stds.append(np.std(bin_elements))

    # print('Done with ind1 stats')

    return (mu, r_mu, n_mu), (std, r_std, n_std), (h, r_h, n_h), (ste, r_ste, n_ste), (lb, r_lb, n_lb), (ub, r_ub, n_ub), (bin_stds, r_bin_stds, n_bin_stds)


def process(data, exps, path, args):
    # num_fict_sets = 2
    num_fict_sets = 10000

    for e in sorted(data.keys()):
        measurements = {}
        measurements[e] = {}

        num_exps = len(exps[e].keys())
        print('Processing {} experiments from {}'.format(num_exps, e))

        # repeat to generate fictions datasets
        for _ in tqdm(range(num_fict_sets), desc='Bootstrap [{}]'.format(e)):
            # randomly pick experiments for the fictitious dataset
            sets = np.random.choice(
                list(exps[e].keys()), size=num_exps, replace=True).tolist()

            # store the indices of the original experiments that make up the fictitious set
            idcs = []
            for exp in sets:
                idcs += exps[e][exp]

            for q in quantities_for_eval:
                if q not in measurements[e].keys():
                    measurements[e][q] = {
                        'mus': [],
                        'r_mus': [],
                        'n_mus': [],
                        'stds': [],
                        'r_stds': [],
                        'n_stds': [],
                        'hs': [],
                        'r_hs': [],
                        'n_hs': [],
                        'stes': [],
                        'r_stes': [],
                        'n_stes': [],
                        'lbs': [],
                        'r_lbs': [],
                        'n_lbs': [],
                        'ubs': [],
                        'r_ubs': [],
                        'n_ubs': [],
                        'bin_stds': [],
                        'r_bin_stds': [],
                        'n_bin_stds': [],
                    }
                m, s, h, ste, lb, ub, bstds = measure_quantity(
                    q, data[e], idcs, args)
                measurements[e][q]['mus'].append(m[0])
                measurements[e][q]['stds'].append(s[0])
                measurements[e][q]['hs'].append(h[0])
                measurements[e][q]['stes'].append(ste[0])
                measurements[e][q]['lbs'].append(lb[0])
                measurements[e][q]['ubs'].append(ub[0])
                measurements[e][q]['bin_stds'].append(bstds[0])

                measurements[e][q]['r_mus'].append(m[1])
                measurements[e][q]['r_stds'].append(s[1])
                measurements[e][q]['r_hs'].append(h[1])
                measurements[e][q]['r_stes'].append(ste[1])
                measurements[e][q]['r_lbs'].append(lb[1])
                measurements[e][q]['r_ubs'].append(ub[1])
                measurements[e][q]['r_bin_stds'].append(bstds[1])

                measurements[e][q]['n_mus'].append(m[2])
                measurements[e][q]['n_stds'].append(s[2])
                measurements[e][q]['n_hs'].append(h[2])
                measurements[e][q]['n_stes'].append(ste[2])
                measurements[e][q]['n_lbs'].append(lb[2])
                measurements[e][q]['n_ubs'].append(ub[2])
                measurements[e][q]['n_bin_stds'].append(bstds[2])

                # print('\n {} done'.format(q))

        # dunmp all measurements in file
        with open(path + '/qstat-{}.pkl'.format(e), 'wb') as h:
            pickle.dump(measurements, h, protocol=pickle.HIGHEST_PROTOCOL)


def plot(exp_files, path, args):
    data = {}
    exps = {}
    join_exps = {
        '1_Experiment': {
            'exp_2': 'exp_1',
            'exp_10': 'exp_9',
            'exp_13': 'exp_12',
            'exp_14': 'exp_12',
        },

        '2_Simu': {},

        '3_Robot': {
            'exp_2': 'exp_1',
            'exp_3': 'exp_1',
            'exp_6': 'exp_5',
            'exp_8': 'exp_7',
            'exp_10': 'exp_9',
            'exp_12': 'exp_11',
            'exp_14': 'exp_13',
            'exp_16': 'exp_15',
            'exp_18': 'exp_17',
            'exp_20': 'exp_19',
            'exp_22': 'exp_21',
            'exp_24': 'exp_23',
            'exp_25': 'exp_23',
        },
    }

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

        # make sure to force filenames to be in order w.r.t. to different experiments
        # this is important for the statistical analysis
        def extract_number(fname):
            exp = fname.split('/')[-1].split('-')[0]
            match = re.search(r'\d+', exp)
            return int(match.group()) if match else 0
        pos = sorted(pos, key=extract_number)

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
        exps[e] = {}

        if args.robot:
            data[e]['ridx'] = []

        for exp_num, p in enumerate(pos):
            exp = p.split('/')[-1].split('-')[0]
            if exp in join_exps[e].keys():
                # if an experiment is split in the data files for storage, here we are handling it as a single experiment again
                exp = join_exps[e][exp]

            # make sure to keep track of which files correspond to which experiment
            if exp not in exps[e].keys():
                exps[e][exp] = []
            exps[e][exp].append(exp_num)

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

    process(data, exps, path, args)
