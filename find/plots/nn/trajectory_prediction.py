#!/usr/bin/env python
import os
import glob
import argparse
from tqdm import tqdm

import colorsys
import matplotlib
import matplotlib.colors as mc
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import find.plots as fp
from find.plots.common import *
from find.utils.features import Velocities
from find.models.storage import ModelStorage
from find.simulation.fish_simulation import FishSimulation
from find.simulation.simulation_factory import available_functors, SimulationFactory, closest_individual
from find.simulation.tf_nn_functors import get_most_influential_individual
from find.plots.spatial.grid_occupancy import construct_grid

from PIL import Image

oc = mpl.path.Path([(0, 0), (1, 0)])

handles_a = [
    mlines.Line2D([0], [0], color='black', marker=oc,
                  markersize=6, label='Median'),
    mlines.Line2D([], [], linestyle='none', color='black', marker='H',
                  markersize=4, label='Mean'),
    mlines.Line2D([], [], linestyle='none', markeredgewidth=1, marker='o',
                  color='black', markeredgecolor='w', markerfacecolor='black', alpha=0.6,
                  markersize=5, label='Sample'),
    mlines.Line2D([], [], linestyle='none', markeredgewidth=0, marker='*',
                  color='black', markeredgecolor='w', markerfacecolor='black',
                  markersize=6, label='Statistical significance'),
]


def simulate(data, model, args):
    simu_factory = SimulationFactory()
    simu = simu_factory(
        data, model, args.nn_functor,  args.backend, args)
    if simu is None:
        import warnings
        warnings.warn('Skipping small simulation')
        return False, None
    if args.backend == 'keras':
        means = np.empty([0, 4])
        stds = np.empty([0, 4])
        for i in range(simu.get_num_iterations()):
            simu.spin_once()
            simu.dump()
            inds = simu.get_individuals()
            m = np.hstack([*inds[0].get_functor().get_means()]).reshape(1, -1)
            means = np.vstack([means, m])
            s = np.hstack([*inds[0].get_functor().get_stds()]).reshape(1, -1)
            stds = np.vstack([stds, s])
        gen_traj = simu.get_stats()[0].get()[args.num_timesteps:, :]
        return True, gen_traj, means, stds
    elif args.backend == 'trajnet':
        simu.spin_once()
        simu.dump()
        inds = simu.get_individuals()
        means = np.hstack([*inds[0].get_functor().get_means()])
        stds = np.hstack([*inds[0].get_functor().get_stds()])
        gen_traj = np.hstack([*inds[0].get_functor().get_full_pred()])
        return True, gen_traj, means, stds
    return False, None, None, None


def generate_traj(exp_files, path, args):
    arg_dict = vars(args)
    arg_dict['iterations'] = args.pred_len
    arg_dict['num_extra_virtu'] = 0
    arg_dict['most_influential_individual'] = 'closest'
    arg_dict['simu_stat_dump_period'] = 1
    arg_dict['distance_inputs'] = True
    arg_dict['stats_enabled'] = True
    args.simu_out_dir = args.path + '/trajectory_pred/' + args.backend
    args.exclude_index = -1

    ms = ModelStorage(path=args.path, create_dirs=False)
    model = ms.load_model(args.path + '/model_checkpoint/' +
                          args.nn_model_ref, args.backend, args)

    skipped_files = 0
    files = glob.glob(args.path + '/' + exp_files['Real'])
    for f in files:
        arg_dict['reference'] = f

        trajectories = np.loadtxt(f)
        if args.pred_len + args.num_timesteps > trajectories.shape[0]:
            skipped_files += 1
            continue

        iters = trajectories.shape[0] - \
            (args.pred_len + args.num_timesteps)
        for i in range(iters):
            args.reference = args.reference.replace(
                'positions', 'positions_{}-{}'.format(args.num_timesteps + i, i + args.num_timesteps + args.pred_len))

            ot = trajectories[i:(i + args.pred_len +
                                 args.num_timesteps), :]

            simu_ok = False
            t = ot.copy()
            t[-args.pred_len:, :] = t[args.num_timesteps, :]
            simu_ok, gen_t, means, stds = simulate(t, model, args)

            if simu_ok:
                gt_fname = args.reference.replace(
                    '.dat', '_gt.dat').replace('raw', 'trajectory_pred/' + args.backend)
                np.savetxt(gt_fname, ot)
                np.savetxt(gt_fname.replace('_gt.dat', '_means.dat'), means)
                np.savetxt(gt_fname.replace('_gt.dat', '_stds.dat'), stds)
                np.savetxt(gt_fname.replace('_gt.dat', '_pred.dat'), gen_t)

                diff_1 = np.linalg.norm(
                    ot[args.num_timesteps:, :2] - gen_t[:, :2], axis=1)
                diff_2 = np.linalg.norm(
                    ot[args.num_timesteps:, 2:] - gen_t[:, 2:], axis=1)
                diff = np.vstack([diff_1, diff_2])
                diff_fname = gt_fname.replace('gt', 'norm')
                np.savetxt(diff_fname, diff.T)
            args.reference = f


def lighten_color(color, amount=0.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def bplot(data, ax, ticks=False):
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 10}
    sns.set_context("paper", rc=paper_rc)

    ax = sns.boxplot(data=data, width=0.09, notch=False,
                     saturation=1, linewidth=1.0, ax=ax,
                     #  whis=[5, 95],
                     showfliers=False,
                     palette=["#e74c3c", "#3498db"]
                     )

    # for i, artist in enumerate(ax.artists):
    #     col = lighten_color(artist.get_facecolor(), 1.2)
    #     artist.set_edgecolor(col)
    #     for j in range(i*6, i*6+6):
    #         line = ax.lines[j]
    #         line.set_color(col)
    #         line.set_mfc(col)
    #         line.set_mec(col)
    #         line.set_linewidth(0.5)

    means = []
    for d in data:
        means.append([np.nanmean(list(d))])
    sns.swarmplot(data=means, palette=['#000000'] * 10,
                  marker='H', size=5, ax=ax)

def plot_pred_accuracy(traj_path, exp_files, path, args):
    files = glob.glob(traj_path + '**/*_norm.dat')

    data = {}
    directories = []
    for f in files:
        directory = f.split('/')[-2]
        norm = np.loadtxt(f) * args.radius

        if directory not in data.keys():
            directories.append(directory)
            data[directory] = {}
            data[directory]['focal'] = {}
            for i in range(args.pred_len):
                data[directory]['focal'][i] = []
                for j in range(1, norm.shape[1]):
                    if 'neigh{}'.format(j) not in data[directory].keys():
                        data[directory]['neigh{}'.format(j)] = {}
                    data[directory]['neigh{}'.format(j)][i] = []

        for m in range(norm.shape[0]):
            data[directory]['focal'][m].append(norm[m, 0])
            for n in range(1, norm.shape[1]):
                data[directory]['neigh{}'.format(n)][m].append(norm[m, n])

    focals = []
    neighs = []
    labels_f = []
    labels_n = []
    for m in range(args.pred_len):
        for key in directories:
            focals.append(data[key]['focal'][m])
            labels_f.append('{} step {}'.format(key, m+1))
            for n in range(1, norm.shape[1]):
                neighs.append(data[key]['neigh{}'.format(n)][m])
                labels_n.append('{} step {}'.format(key, m+1))

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    bplot(focals, ax)
    ax.set_ylabel('Trajectory deviation (m)')
    ax.set_xlabel('')
    ax.set_xticklabels(labels_f)
    plt.xticks(rotation=45)
    plt.savefig(traj_path + 'focals.png', dpi=300)

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    bplot(neighs, ax)
    ax.set_ylabel('Trajectory deviation (m)')
    ax.set_xlabel('')
    ax.set_xticklabels(labels_n)
    plt.xticks(rotation=45)
    plt.savefig(traj_path + 'neighs.png', dpi=300)


def fish_image_on_axis(pictures, vel, traj, ax, args):
    phi = np.arctan2(vel[args.num_timesteps-1, 1],
                        vel[args.num_timesteps-1, 0]) * 180 / np.pi
    rimage = pictures[0].rotate(phi)
    size = 2.7
    ax.imshow(rimage, extent=[
        traj[args.num_timesteps-1, 0] - size, traj[args.num_timesteps-1, 0] + size, 
        traj[args.num_timesteps-1, 1] - size, traj[args.num_timesteps-1, 1] + size], aspect='equal', zorder=10)

    for n in range(1, traj.shape[1] // 2):
        phi = np.arctan2(vel[args.num_timesteps-1, n * 2 + 1],
                            vel[args.num_timesteps-1, n * 2]) * 180 / np.pi
        rimage = pictures[1].rotate(phi)
        ax.imshow(rimage , extent=[
            traj[args.num_timesteps-1, n * 2] - size, traj[args.num_timesteps-1, n * 2] + size, 
            traj[args.num_timesteps-1, n * 2 + 1] - size, traj[args.num_timesteps-1, n * 2 + 1] + size], aspect='equal', zorder=10)     


def plot_fish_pred_cone(traj_path, exp_files, path, args):
    dirs = []
    files = glob.glob(traj_path + '*')
    for f in files:
        if os.path.isdir(f):
            dirs.append(f.split('/')[-1])

    files = glob.glob(traj_path + '{}/*_pred.dat'.format(dirs[0]))

    mpath = os.path.dirname(fp.__file__)
    pictures = [Image.open(
        mpath + '/res/fish_red_nicer.png')
        # .resize((45, 45))
        ,
        Image.open(
        mpath + '/res/fish_blue_nicer.png')
        # .resize((45, 45)),
    ]

    data = {}
    for fno, f in enumerate(files):
        gtruth = np.loadtxt(f.replace('_pred.dat', '_gt.dat')) * args.radius
        vel = Velocities([gtruth], args.timestep).get()[0]
        
        abort = False
        data = {}
        for d in dirs:
            fcomp = f.replace(dirs[0], d)
            if not os.path.exists(fcomp):
                abort = True
                break

            data[d] = {
                'pred': np.loadtxt(fcomp) * args.radius,
                'means': np.loadtxt(fcomp.replace('_pred.dat', '_means.dat')) * args.radius,
                'stds': np.loadtxt(fcomp.replace('_pred.dat', '_stds.dat')) * args.radius,
            }

        if abort:
            continue

        fig = plt.figure()
        fig.set_figwidth(10)
        fig.set_figheight(8)
        gs0 = fig.add_gridspec(1, 2)
        gs0.set_width_ratios([2, 1])
        gs1 = gs0[0, 1].subgridspec(2, 1)

        ax = fig.add_subplot(gs0[0, 0])
        iax_r = fig.add_subplot(gs1[0, 0]) 
        iax_l = fig.add_subplot(gs1[1, 0])

        # ground truth
        axes = [ax, iax_r, iax_l] # ! this is only limited to 2 models for now
        obs_len = args.num_timesteps

        ax = sns.lineplot(x=gtruth[(obs_len-1):, 0], y=gtruth[(obs_len-1):, 1], label='Ground truth (prediction)', ax=ax, marker='o', linestyle=':', color='black', zorder=100, markersize=4)
        ax = sns.lineplot(x=gtruth[:obs_len, 0], y=gtruth[:obs_len, 1], label='Ground truth (Observation)', ax=ax, marker='o', linestyle='--', color='black', zorder=100, markersize=4)

        for n in range(1, gtruth.shape[1] // 2):
            ax = sns.lineplot(x=gtruth[(obs_len-1):, n * 2], y=gtruth[(obs_len-1):, n * 2 + 1], ax=ax, marker='o', linestyle=':', color='black', zorder=100, markersize=4)
            ax = sns.lineplot(x=gtruth[:obs_len, n * 2], y=gtruth[:obs_len, n * 2 + 1], ax=ax, marker='o', linestyle='--', color='black', zorder=100, markersize=4)

        # fish images
        for cax in axes:
            fish_image_on_axis(pictures, vel, gtruth, cax, args)

        preds = []
        lcolours = sns.color_palette("hls", len(data))
        for mnum, (model, meas) in enumerate(data.items()):
            pred = meas['pred']
            means = meas['means']
            stds = meas['stds']
            
            model_label = model
            if model_label == 'keras':
                model_label = 'find'

            xs = []
            ys = []
            for i in range(means.shape[0]):
                xs += np.random.normal(means[i, 0], stds[i, 0], 5000).tolist()
                ys += np.random.normal(means[i, 1], stds[i, 1], 5000).tolist()
            traj = np.array([xs, ys]).T

            sub_data = {}
            sub_data[model_label] = [traj]
            x, y, z = construct_grid(sub_data, model_label, args, 5)
            palette = sns.color_palette('viridis', args.grid_bins)
            cmap = ListedColormap(palette)
            c = axes[mnum+1].pcolormesh(x, y, z, cmap=cmap, shading='auto',
                          vmin=0.0, vmax=np.max(z), alpha=0.9)

            # trajectories
            ax = sns.lineplot(x=pred[:, 0], y=pred[:, 1], label='Prediction ({})'.format(model_label), 
            ax=ax, marker='o', linestyle=':', color=lcolours[mnum], zorder=100, markersize=4)

            ax = sns.lineplot(x=pred[:, 2], y=pred[:, 3], ax=ax, marker='o', linestyle=':', color=lcolours[mnum], zorder=100, markersize=4)

            axes[mnum+1] = sns.lineplot(x=pred[:, 0], y=pred[:, 1], ax=axes[mnum+1], marker='o', linestyle=':', color=lcolours[mnum], zorder=100, markersize=4)

            axes[mnum+1].set_title(model_label)
            preds.append(pred)

        # axes 
        xys = np.empty((0, 2))
        for n in range(gtruth.shape[1] // 2):
            xys = np.vstack([xys, gtruth[:, (n*2):(n*2+2)]])

        for p in preds:
            for n in range(p.shape[1] // 2):
                xys = np.vstack([xys, p[:, (n*2):(n*2+2)]])

        mins = np.min(xys, axis=0)
        maxs = np.max(xys, axis=0)

        outer = plt.Circle((0, 0), args.radius * 1.0005,
                        color='black', fill=False)
        ax.add_artist(outer)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        ax.set_xlim([mins[0] - 2, maxs[0] + 2])
        ax.set_ylim([mins[1] - 2, maxs[1] + 2])
        
        for idx, pred in enumerate(preds):
            xys = np.empty((0, 2))
            xys = np.vstack([xys, gtruth[-args.pred_len-2:, :2]])
            xys = np.vstack([xys, pred[:, :2]])
            mins = np.min(xys, axis=0)
            maxs = np.max(xys, axis=0)

            outer = plt.Circle((0, 0), args.radius * 1.0005,
                            color='black', fill=False)
            axes[idx+1].add_artist(outer)
            axes[idx+1].set_aspect('equal', 'box')
            axes[idx+1].set_xlim([mins[0] - 1, maxs[0] + 1])
            axes[idx+1].set_ylim([mins[1] - 1, maxs[1] + 1])

        plt.savefig(traj_path + 'trajectory_{}.png'.format(fno), dpi=300)
        plt.close()

def plot(exp_files, path, args):
    traj_path = args.path + '/trajectory_pred/'
    if args.force_regenerate or not os.path.isdir(traj_path + args.backend):
        generate_traj(exp_files, path, args)

    plot_pred_accuracy(traj_path, exp_files, path, args)
    plot_fish_pred_cone(traj_path, exp_files, path, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot trajectory predictions given a trained model and the ground truth file(s)')
    parser.add_argument('--nn_model_ref',
                        type=str,
                        help='Model to consider as reference for its parameters',
                        default='best_model.h5',
                        required=False)
    parser.add_argument('--type',
                        nargs='+',
                        default=['Real', 'Hybrid', 'Virtual'],
                        choices=['Real', 'Hybrid', 'Virtual'])
    parser.add_argument('--original_files',
                        type=str,
                        default='raw/*processed_positions.dat',
                        required=False)
    parser.add_argument('--radius',
                        type=float,
                        help='Radius',
                        default=0.25,
                        required=False)
    parser.add_argument('--center',
                        type=float,
                        nargs='+',
                        help='The centroidal coordinates for the setups used',
                        default=[0.0, 0.0],
                        required=False)
    parser.add_argument('--num_timesteps',
                        type=int,
                        help='Observation length for the model',
                        default=5,
                        required=False)
    parser.add_argument('--pred_len',
                        type=int,
                        help='Prediction length for the model (Depending on the model, multiple single predictions might be made instead)',
                        default=1,
                        required=False)
    parser.add_argument('--nn_functor',
                        default=available_functors()[0],
                        choices=available_functors())
    parser.add_argument('--var_coef', type=float,
                        help='Prediction variance coefficient',
                        default=1.0,
                        required=False)
    parser.add_argument('--force_regenerate',
                        action='store_true',
                        help='Regenerate trajectory predictions',
                        default=False,
                        required=False)
    args = parser.parse_args()

    exp_files = {}
    for t in args.type:
        if t == 'Real':
            exp_files[t] = args.original_files

    plot(exp_files, './', args)
