#!/usr/bin/env python
import os
import gc
import sys
import glob
import time
import random
import signal
import argparse
import subprocess
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

import find.plots as fp
from find.utils.features import Velocities
from find.utils.utils import compute_leadership
# from find.plots.common import *

TOULOUSE_DATA = False
TOULOUSE_CPP_DATA = False

first_sigint = True
ffmpeg_command = 'echo foo'


def draw_single_frame(traj, vel, i, fps, pictures, out_dir, args):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()

    # draw each individual's image
    for inum, j in enumerate(range(traj.shape[1] // 2)):
        x = traj[0, j * 2]
        y = traj[0, j * 2 + 1]

        if not args.fish_like:
            plt.scatter(x, y, marker='.',
                        label='Individual ' + str(inum) + ' ' + "{:.2f}".format(x) + ' ' + "{:.2f}".format(y))
            plt.quiver(
                x, y, vel[0, j * 2], vel[0, j * 2 + 1], scale=1, units='xy')
        else:
            beat_frames = int(args.tail_period / 2. * fps)
            if i % beat_frames:
                tail_beat = True
            else:
                tail_beat = False

            phi = np.arctan2(vel[0, j * 2 + 1], vel[0, j * 2]) * 180 / np.pi

            # if it's time to kick then flip the image
            if tail_beat:
                rimage = pictures[j][0].rotate(phi)
            else:
                rimage = pictures[j][1].rotate(phi)

            ax.imshow(rimage, extent=[x - args.body_len, x + args.body_len, y -
                                      args.body_len, y + args.body_len], aspect='equal')

    # plot the circular arena TODO: this should be more generic
    if args.dark:
        color = 'white'
    else:
        color = 'black'
    outer = plt.Circle(
        args.center, args.radius*1.015, color=color, fill=False)
    ax.add_artist(outer)
    ax.axis('off')
    ax.set_xlim([-args.radius*1.05, args.radius*1.05])
    ax.set_ylim([-args.radius*1.05, args.radius*1.05])
    plt.tight_layout()

    # save image
    png_fname = Path(out_dir).joinpath(str(i).zfill(6))
    if args.range:
        png_fname = Path(out_dir).joinpath(str(args.range[0] + i).zfill(6))
    plt.savefig(
        str(png_fname) + '.png',
        transparent=True,
        dpi=300
    )
    # try to handle memory leaks
    plt.close('all')
    gc.collect()


def plot(foo, path, args):
    global first_sigint
    global ffmpeg_command

    mpath = os.path.dirname(fp.__file__)

    # random select an experiment to visualise
    trajectories = []
    if args.traj_visualisation_list == 'random':
        files = glob.glob(args.path + '/generated/*_positions.dat')
        trajectories.append(np.loadtxt(random.choice(files)) * args.radius)
    else:
        files = args.traj_visualisation_list
        for f in files:
            if TOULOUSE_DATA:
                fi = open(f)
                # to allow for loading fortran's doubles
                strarray = fi.read().replace("D+", "E+").replace("D-", "E-")
                fi.close()
                num_ind = len(strarray.split('\n')[0].strip().split('  '))
                positions = np.fromstring(
                    strarray, sep='\n').reshape(-1, num_ind)
            elif TOULOUSE_CPP_DATA:
                positions = np.loadtxt(f)[:, 2:]
            else:
                positions = np.loadtxt(f)

            # if a value is provided then assume data is in pixels and adjust the input trajectories
            if args.pix2m > 0:
                positions = positions * args.pix2m
                num_inds = positions.shape[1] // 2
                for idx in range(num_inds):
                    positions[:, idx*2] = positions[:, idx*2] - 0.33
                    positions[:, idx*2+1] = positions[:, idx*2+1] - 0.33
            else:
                positions *= args.radius

            trajectories.append(positions)

    # repeat the visualization for a list of given trajectories
    for fidx, traj in enumerate(trajectories):
        vel = Velocities([traj], args.timestep).get()[0]

        if args.fish_like:
            pictures = {}

            if traj.shape[1] // 2 == 2:
                pictures[0] = []
                pictures[1] = []

                pictures[0].append(Image.open(
                    str(Path(mpath).joinpath('res').joinpath('fish_artwork_red_down.png'))))
                pictures[0].append(Image.open(
                    str(Path(mpath).joinpath('res').joinpath('fish_artwork_red_up.png'))))

                pictures[1].append(Image.open(
                    str(Path(mpath).joinpath('res').joinpath('fish_artwork_blue_down.png'))))
                pictures[1].append(Image.open(
                    str(Path(mpath).joinpath('res').joinpath('fish_artwork_blue_up.png'))))
            else:
                for ind in range(traj.shape[1] // 2):
                    pictures[ind] = []
                    pictures[ind].append(Image.open(
                        str(Path(mpath).joinpath('res').joinpath('fish_artwork_blue_down.png'))))
                    pictures[ind].append(Image.open(
                        str(Path(mpath).joinpath('res').joinpath('fish_artwork_blue_up.png'))))

        # pick the range of trajectories to visualise
        if args.range is not None:  # keep the timesteps defined by the CLI parameters
            idcs = list(map(int, args.range))
            traj = traj[idcs[0]:idcs[1], :]
            vel = vel[idcs[0]:idcs[1], :]

        fps = 1 // args.timestep
        # In case the user wants to produce smoother videos they can opt to fill frames between actual data points
        if args.fill_between > 0:
            fps += (fps - 1) * args.fill_between

            filled_traj = np.empty(
                ((traj.shape[0] - 1) * args.fill_between, 0))
            filled_vel = np.empty(((traj.shape[0] - 1) * args.fill_between, 0))
            for idx in range(traj.shape[1] // 2):
                ft = np.empty((0, 2))
                fv = np.empty((0, 2))
                for i in tqdm(range(traj.shape[0] - 1), desc='filling trajectories'):
                    fill_x = np.linspace(
                        traj[i, idx * 2], traj[i + 1, idx * 2], args.fill_between)
                    fill_y = np.linspace(
                        traj[i, idx * 2 + 1], traj[i + 1, idx * 2 + 1], args.fill_between)
                    fill_vx = np.linspace(
                        vel[i, idx * 2], vel[i + 1, idx * 2], args.fill_between)
                    fill_vy = np.linspace(
                        vel[i, idx * 2 + 1], vel[i + 1, idx * 2 + 1], args.fill_between)
                    ft = np.vstack(
                        (ft, np.vstack((fill_x, fill_y)).T))
                    fv = np.vstack(
                        (fv, np.vstack((fill_vx, fill_vy)).T))
                filled_traj = np.hstack((filled_traj, ft))
                filled_vel = np.hstack((filled_vel, fv))
            traj = np.vstack((filled_traj, traj[-1, :]))
            vel = np.vstack((filled_vel, vel[-1, :]))

        print('Trajectories at {} fps'.format(fps))

        # construct output dir paths
        folder_name = os.path.basename(files[fidx]).split('.')[0]
        out_dir = str(Path(path).joinpath(folder_name))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # start a thread pool to handle frames in parallel
        num_threads = args.num_plot_threads
        if num_threads > 1:
            pool = Pool(num_threads)

        ffmpeg_command = "cd {}; ffmpeg -y -framerate {} -pattern_type glob -i '*.png'  -c:v libx264 -pix_fmt yuv420p -r {} ../{}.mp4".format(
            out_dir, fps, fps, folder_name)

        # setup signal handler
        def signal_handler(sig, frame):
            global first_sigint
            global ffmpeg_command

            print('SIGINT caught...')
            if args.w_mp4 and first_sigint:
                first_sigint = False
                subprocess.call(ffmpeg_command, shell=True)
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)

        # iterate over all timesteps
        tsteps = traj.shape[0]
        for i in tqdm(range(0, tsteps-num_threads, num_threads)):
            # start perf timer for the saving section
            # tic = time.perf_counter()

            if num_threads == 1:
                draw_single_frame(
                    traj[i, :].reshape(-1, traj.shape[1]),
                    vel[i, :].reshape(-1, traj.shape[1]), i,
                    fps, pictures, out_dir, args)
            else:
                pool_args = [
                    [
                        traj[idx, :].reshape(-1, traj.shape[1]),
                        vel[idx, :].reshape(-1, traj.shape[1]), idx,
                        fps, pictures, out_dir, args
                    ] for idx in range(i, i+num_threads)
                ]
                pool.starmap(draw_single_frame, pool_args)

            # stop perf timer for the saving section
            # toc = time.perf_counter()
            # print('\n{} workers done in ({:.4f} s)'.format(
            #     num_threads, toc - tic))

        if args.w_mp4:
            subprocess.call(ffmpeg_command, shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise the positions of the fish accompanied by the feature information')
    parser.add_argument('--traj_visualisation_list',
                        type=str,
                        nargs='+',
                        help='List of files to visualise',
                        default='random',
                        required=False)
    parser.add_argument('--fish_like', action='store_true',
                        help='Images instead of points',
                        default=False)
    parser.add_argument('--turing', action='store_true',
                        help='Same image for all individuals to perform a turing test',
                        default=False)
    parser.add_argument('--info', action='store_true',
                        help='Display info',
                        default=False)
    parser.add_argument('--dark', action='store_true',
                        help='Render dark friendly icons',
                        default=False)
    parser.add_argument('--exclude_index', '-e', type=int,
                        help='Index of the virtual individual',
                        required=False,
                        default=-1)
    parser.add_argument('--range', nargs='+',
                        help='Vector containing the start and end index of trajectories to be plotted',
                        required=False)
    parser.add_argument('--radius', '-r', type=float,
                        help='Radius',
                        default=0.25,
                        required=False)
    parser.add_argument('--timestep', '-t', type=float,
                        help='Simulation timestep',
                        required=True)
    parser.add_argument('--fill_between', type=int,
                        help='Fill frames between timesteps',
                        default=0,
                        required=False)
    parser.add_argument('--center',
                        type=float,
                        nargs='+',
                        help='The centroidal coordinates for the setups used',
                        default=[0.0, 0.0],
                        required=False)
    parser.add_argument('--tail_period',
                        type=float,
                        help='Tail frequency to change the image of the fish (only used in fish_like)',
                        default=0.5,
                        required=False)
    parser.add_argument('--pix2m',
                        type=float,
                        help='pixel to meter coefficient',
                        default=-1,
                        required=False)
    parser.add_argument('--body_len', type=float,
                        help='Body length of the individuals (for fish)',
                        default=0.035,
                        required=False)
    args = parser.parse_args()

    plot(None, './', args)
