#!/usr/bin/env python

import os
import argparse
from tqdm import tqdm

import find.plots.nn as nn
import find.plots.spatial as sp
import find.plots.correlation as co
import find.plots.trajectory_visualisation as vi
import find.plots.dl_si_2021 as dl_si_2021

from find.simulation.simulation_factory import available_functors


def plot_selector(key):
    if key in sp.available_plots():
        return sp.get_plot(p), sp.source
    elif key in vi.available_plots():
        return vi.get_plot(p), vi.source
    elif key in nn.available_plots():
        return nn.get_plot(p), nn.source
    elif key in co.available_plots():
        return co.get_plot(p), co.source
    elif key in dl_si_2021.available_plots():
        return dl_si_2021.get_plot(p), dl_si_2021.source
    else:
        assert False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot manager')
    parser.add_argument('--path', '-p',
                        type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--timestep', '-t',
                        type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--timesteps_skip',
                        type=int,
                        help='Timesteps skipped between input and prediction',
                        default=0,
                        required=False)

    # available plots
    plot_list = sp.available_plots() + vi.available_plots() + \
        co.available_plots() + nn.available_plots() + dl_si_2021.available_plots()

    plot_conf = parser.add_argument_group('Plot configuration')
    plot_conf.add_argument('--plot',
                           nargs="+",
                           default='all_spatial_and_correllation',
                           choices=plot_list + ['all_spatial_and_correllation'])
    plot_conf.add_argument('--plot_out_dir', type=str,
                           help='Directory for plot output files (always relative to the experiment path)',
                           default='plots',
                           required=False)
    plot_conf.add_argument('--type',
                           nargs='+',
                           default=['Real', 'Hybrid', 'Virtual'],
                           choices=['Real', 'Hybrid', 'Virtual'])
    plot_conf.add_argument('--original_files',
                           type=str,
                           default='raw/*processed_positions.dat',
                           required=False)
    plot_conf.add_argument('--hybrid_files',
                           type=str,
                           default='generated/*generated_positions.dat',
                           required=False)
    plot_conf.add_argument('--virtual_files',
                           type=str,
                           default='generated/*generated_virtu_positions.dat',
                           required=False)
    plot_conf.add_argument('--num_virtual_samples',
                           type=int,
                           help='Number of samples to use when computing metrics for the virtual data',
                           default=-1,
                           required=False)

    spatial_options = parser.add_argument_group('Spatial plot options')
    spatial_options.add_argument('--radius', '-r',
                                 type=float,
                                 help='Radius',
                                 default=0.25,
                                 required=False)
    spatial_options.add_argument('--grid_bins',
                                 type=int,
                                 help='Number of bins for the occupancy grid plot',
                                 default=416,
                                 required=False)
    spatial_options.add_argument('--grid_smooth',
                                 action='store_true',
                                 help='Smooth the grid for visual reasons if true',
                                 default=False,
                                 required=False)
    spatial_options.add_argument('--grid_cutoff_thres',
                                 type=float,
                                 help='Cutoff point threshold for the percentage of points that are allowed to be removed to not squash the grid drawing colours',
                                 default=0.05,
                                 required=False)
    spatial_options.add_argument('--grid_cutoff_val',
                                 type=float,
                                 help='Force the cutoff value of the grid for consistency (overrides grid_cutoff_thres)',
                                 default=-1,
                                 required=False)
    spatial_options.add_argument('--kde_gridsize',
                                 type=int,
                                 help='Grid size for kernel density estimation plots',
                                 default=5000,
                                 required=False)
    spatial_options.add_argument('--center',
                                 type=float,
                                 nargs='+',
                                 help='The centroidal coordinates for the setups used',
                                 default=[0.0, 0.0],
                                 required=False)
    spatial_options.add_argument('--prediction_len', type=int,
                                 help='Predictions to plot',
                                 default=5,
                                 required=False)
    spatial_options.add_argument('--observation_len', type=int,
                                 help='Observations to plot',
                                 default=0,
                                 required=False)
    spatial_options.add_argument('--radius_grid_res',
                                 type=int,
                                 help='Resolution (in m) for the radius of the focal individual in the future trajectory variance plot',
                                 default=0.025,
                                 required=False)
    spatial_options.add_argument('--angle_grid_res',
                                 type=int,
                                 help='Resolution (in deg) for the angle to the wall of the focal individual future trajectory variance plot',
                                 default=5,
                                 required=False)
    spatial_options.add_argument('--interdist_grid_res',
                                 type=float,
                                 help='Resolution (in m) for the interinidividual distance in the future trajectory variance plot',
                                 default=0.025,
                                 required=False)
    spatial_options.add_argument('--viewing_angle_grid_res',
                                 type=float,
                                 help='Resolution (in degr) for the interinidividual distance in the future trajectory variance plot',
                                 default=45,
                                 required=False)

    cor_plot = parser.add_argument_group('Correlation plot options')
    cor_plot.add_argument('--tcor',
                          type=float,
                          default=25.0,
                          help='Time window to consider when computing correlation metrics',
                          required=False)
    cor_plot.add_argument('--ntcor',
                          type=int,
                          default=1,
                          help='Number of timesteps to includ in the correlation metrics computaion',
                          required=False)

    traj_options = parser.add_argument_group(
        'Trajectory visualisation plot options')
    traj_options.add_argument('--traj_visualisation_list',
                              type=str,
                              nargs='+',
                              help='List of files to visualise',
                              default='random',
                              required=False)
    traj_options.add_argument('--open', action='store_true',
                              help='Visualize the open setup', default=False)
    traj_options.add_argument('--fish_like', action='store_true',
                              help='Images instead of points',
                              default=False)
    traj_options.add_argument('--turing', action='store_true',
                              help='Same image for all individuals to perform a turing test',
                              default=False)
    traj_options.add_argument('--info', action='store_true',
                              help='Display info',
                              default=False)
    traj_options.add_argument('--dark', action='store_true',
                              help='Render dark friendly icons',
                              default=False)
    traj_options.add_argument('--exclude_index', '-e', type=int,
                              help='Index of the virtual individual',
                              required=False,
                              default=-1)
    traj_options.add_argument('--range', nargs='+',
                              type=int,
                              help='Vector containing the start and end index of trajectories to be plotted',
                              required=False)
    traj_options.add_argument('--dpi', type=int,
                              help='Radius',
                              default=300,
                              required=False)
    traj_options.add_argument('--fill_between', type=int,
                              help='Fill frames between timesteps',
                              default=0,
                              required=False)

    nn_options = parser.add_argument_group(
        'NN training history visualisation optioins')
    nn_options.add_argument('--nn_compare_dirs',
                            type=str,
                            nargs='+',
                            help='List of directories to look through and analyse',
                            required=False)
    nn_options.add_argument('--nn_compare_out_dir',
                            type=str,
                            help='Directory to output NN analysis results',
                            default='nn_comparison',
                            required=False)
    nn_options.add_argument('--nn_delimiter',
                            type=str,
                            help='Delimiter used in the log files',
                            default=',',
                            required=False)
    nn_options.add_argument('--nn_last_epoch',
                            type=int,
                            help='Plot up to nn_last_epoch data points. -1 stands for all, -2 stands for up to the min of iterations across the experiments',
                            default=-1,
                            required=False)
    nn_options.add_argument('--nn_num_legend_parents',
                            type=int,
                            help='Number of parent directories to show in the legend',
                            default=1,
                            required=False)
    nn_options.add_argument('--nn_num_sample_epochs',
                            type=int,
                            help='Number of samples to plot. -1 will consider all available points',
                            default=-1,
                            required=False)
    nn_options.add_argument('--nn_model_ref',
                            type=str,
                            help='Model to consider as reference for its parameters',
                            default='best_model.h5',
                            required=False)
    nn_options.add_argument('--backend',
                            help='Backend selection',
                            default='keras',
                            choices=['keras', 'trajnet'])
    nn_options.add_argument('--num_timesteps',
                            type=int,
                            help='Observation length for the model',
                            default=5,
                            required=False)
    nn_options.add_argument('--pred_len',
                            type=int,
                            help='Prediction length for the model (Depending on the model, multiple single predictions might be made instead)',
                            default=1,
                            required=False)
    nn_options.add_argument('--nn_functor',
                            default=available_functors()[0],
                            choices=available_functors())
    nn_options.add_argument('--var_coef', type=float,
                            help='Prediction variance coefficient',
                            default=1.0,
                            required=False)
    nn_options.add_argument('--force_regenerate',
                            action='store_true',
                            help='Regenerate trajectory predictions',
                            default=False,
                            required=False)
    args = parser.parse_args()
    args.timestep = args.timestep * (args.timesteps_skip + 1)
    args.plot_out_dir = args.path + '/' + args.plot_out_dir

    if args.plot == 'all_spatial_and_correllation':
        args.plot = sp.available_plots() + co.available_plots()

    if args.plot == 'all_spatial':
        args.plot = sp.available_plots()

    if args.plot == 'all_correlation':
        args.plot = co.available_plots()

    exp_files = {}
    for t in args.type:
        if t == 'Real':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    if not os.path.exists(args.plot_out_dir):
        os.makedirs(args.plot_out_dir)

    for p in tqdm(args.plot, desc='Plotting the selected quantities ({})'.format(len(args.plot))):
        pfunc, ptype = plot_selector(p)

        if ptype == 'nn':
            outpath = args.nn_compare_out_dir
            if not os.path.exists(outpath):
                os.makedirs(outpath)
        else:
            outpath = args.plot_out_dir + '/' + ptype + '/'
            if not os.path.exists(outpath):
                os.makedirs(outpath)

        pfunc(exp_files, outpath, args)
