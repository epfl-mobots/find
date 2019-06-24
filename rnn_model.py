#!/usr/bin/env python

import glob
import argparse
import numpy as np


def load(exp_path, fname):
    files = glob.glob(exp_path + '/*' + fname)
    data = []
    for f in files:
        matrix = np.loadtxt(f)
        data.append(matrix)
    return data, files


def angle_to_pipi(dif):
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif


def split_polar(data, args={'center' : (0, 0)}):
    if 'center' not in args.keys():
        args['center'] = (0, 0)

    pos = data['pos']
    vel = data['vel']

    inputs = None
    outputs = None
    for p in pos:
        for n in range(p.shape[1] // 2):
            rads = p[:, n*2:n*2+2]
            rads[:, 0] -= args['center'][0]
            rads[:, 1] -= args['center'][1]
            phis = np.arctan2(rads[:, 1], rads[:, 0])
            rads[:, 0] = rads[:, 0] ** 2
            rads[:, 1] = rads[:, 1] ** 2
            rads = np.sqrt(rads[:, 0] + rads[:, 1])

            rads_t_1 = np.roll(rads, shift=-1, axis=0)
            phis_t_1 = np.roll(phis, shift=-1, axis=0)

            drads = rads - rads_t_1
            dphis = np.array(list(map(lambda x: angle_to_pipi(x), rads - rads_t_1)))

            X = np.array([rads_t_1, np.cos(phis_t_1), np.sin(phis_t_1)])
            Y = np.array([drads, np.cos(dphis), np.sin(dphis)])
            if inputs is None:
                inputs = X
                outputs = Y
            else:
                inputs = np.append(inputs, X, axis=1)
                outputs = np.append(outputs, Y, axis=1)
    return inputs, outputs


def split_data(data, split_func=split_polar, args={}):
    return split_func(data, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='RNN model to reproduce fish motion')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    # parser.add_argument('--filename', '-f', type=str,
    #                     help='Position file name',
    #                     required=True)
    # parser.add_argument('--fps', type=int,
    #                     help='Camera framerate',
    #                     required=True)
    # parser.add_argument('--centroids', '-c', type=int,
    #                     help='Frames to use in order to compute the centroidal positions',
    #                     required=True)
    args = parser.parse_args()

    pos, _ = load(args.path, 'positions_filtered.dat')
    vel, _ = load(args.path, 'velocities_filtered.dat')
    data = {
        'pos' : pos,
        'vel' : vel
    }
    X, Y = split_data(data)

    
