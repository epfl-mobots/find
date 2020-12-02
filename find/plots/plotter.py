#!/usr/bin/env python

import os
import argparse
from tqdm import tqdm

import find.plots.spatial as sp
import find.plots.trajectory_visualisation as vi
from find.plots.common import uni_colours


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
    plot_list = sp.available_plots() + vi.available_plots()

    plot_conf = parser.add_argument_group('Plot configuration')
    plot_conf.add_argument('--plot',
                           nargs="+",
                           default='all_spatial',
                           choices=plot_list + ['all_spatial'])
    plot_conf.add_argument('--plot_out_dir', type=str,
                           help='Directory for plot output files (always relative to the experiment path)',
                           default='plots',
                           required=False)
    plot_conf.add_argument('--type',
                           nargs='+',
                           default=['Original', 'Hybrid', 'Virtual'],
                           choices=['Original', 'Hybrid', 'Virtual'])
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

    spatial_options = parser.add_argument_group('Spatial plot options')
    spatial_options.add_argument('--radius',
                                 type=float,
                                 help='Radius',
                                 default=0.25,
                                 required=False)
    spatial_options.add_argument('--grid_bins',
                                 type=int,
                                 help='Number of bins for the occupancy grid plot',
                                 default=300,
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

    args = parser.parse_args()
    args.timestep = args.timestep * (args.timesteps_skip + 1)
    args.plot_out_dir = args.path + '/' + args.plot_out_dir

    if args.plot == 'all_spatial':
        args.plot = sp.available_plots()

    exp_files = {}
    for t in args.type:
        if t == 'Original':
            exp_files[t] = args.original_files
        elif t == 'Hybrid':
            exp_files[t] = args.hybrid_files
        elif t == 'Virtual':
            exp_files[t] = args.virtual_files

    if not os.path.exists(args.plot_out_dir):
        os.makedirs(args.plot_out_dir)

    for p in tqdm(args.plot, desc='Plotting the selected quantities {}'.format(str(args.plot))):
        pfunc, ptype = (
            sp.get_plot(p), sp.source) if p in sp.available_plots() else (vi.get_plot(p), vi.source)

        if not os.path.exists(args.plot_out_dir + '/' + ptype + '/'):
            os.makedirs(args.plot_out_dir + '/' + ptype + '/')

        pfunc(exp_files, args.plot_out_dir + '/' + ptype + '/', args)
