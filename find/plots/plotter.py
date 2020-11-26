#!/usr/bin/env python

import os
import tqdm
import argparse

# from find.models.loader import Loader
# from find.models.storage import ModelStorage
# from find.models.model_factory import ModelFactory

from find.plots.plot_common_utils import *

import find.plots.spatial as sp
import find.plots.visualisation as vi


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
                           default=plot_list[0],
                           choices=plot_list)
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
                                 help='Raidus',
                                 default=0.25,
                                 required=False)
    args = parser.parse_args()
    args.timestep = args.timestep * (args.timesteps_skip + 1)
    args.plot_out_dir = args.path + '/' + args.plot_out_dir

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

    # TODO: do something else for the colours
    palette = sns.cubehelix_palette(len(exp_files.keys()))
    colours = sns.color_palette(palette)

    for p in args.plot:
        pfunc = sp.get_plot(p) if p in sp.available_plots() else vi.get_plot(p)
        pfunc(exp_files, colours, args.plot_out_dir + '/', args)
