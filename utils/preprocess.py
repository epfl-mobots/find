#!/usr/bin/env python
import argparse
import datetime
import glob
import numpy as np
import os
import socket  # for get hostname
from pathlib import Path
from word2number import w2n
from pprint import pprint

from features import Velocities, Accelerations
from utils import ExperimentInfo, Center, Normalize


class Archive:
    """Serialization class for the fish experiments."""

    def __init__(self, args={'debug': False}):
        """
        :param args: dict, optional of generic arguments for the class
        """
        if args['debug']:
            self._experiment_path = 'test'
        else:
            self._hostname = socket.gethostname()
            self._timestamp = datetime.date.today().strftime('%Y_%m_%d') + '-' + \
                datetime.datetime.now().strftime('%H_%M_%S')
            self._experiment_path = self._hostname + '_' + self._timestamp

        if not os.path.exists(self._experiment_path):
            os.makedirs(self._experiment_path)

    def path(self):
        """
        :return: Path to the experiment folder that was created in the constructor
        """
        return Path(self._experiment_path)

    def save(self, data, filename):
        """
        :param data: np.array of arbitrary numerical data
        :param filename: str filename for the output data file
        """
        if isinstance(data, (np.ndarray, np.generic)):
            np.savetxt(self.path().joinpath(filename), data)
        else:
            assert False, 'Can not store data structures of this type'


def load(exp_path, fname, has_probs=True):
    """
    :param exp_path: str path to the experiment folder where the data we want to load are stored
    :param fname: str the name of the files we want to load
    :return: tuple(list(np.array), list) of the matrices and corresponding file names
    """
    files = glob.glob(exp_path + '/**/' + fname)
    data = []
    for f in files:
        matrix = np.loadtxt(f, skiprows=1)
        if has_probs:
            matrix = np.delete(matrix, np.s_[2::3], 1)
        data.append(matrix)
    return data, files


def preprocess(data, filter_func, args={'scale': 1.0}):
    """
    :param data: list(np.array) of position data for different fish individuals or experiments
    :param filter_func: func that will apply a smoothing on the data
    :param args: dict, optional for extra arguments that need to be passed to this function
    :return: list(np.array), ExperimentInfo
    """
    # every matrix should have the same number of rows
    if 'initial_keep' in args.keys():
        for i in range(len(data)):
            skip = data[i].shape[0] - args['initial_keep']
            data[i] = data[i][skip:, :]
    else:
        min_rows = float('Inf')
        for i in range(len(data)):
            if data[i].shape[0] < min_rows:
                min_rows = data[i].shape[0]
        for i in range(len(data)):
            data[i] = data[i][:min_rows, :]

    # filtering the data with a simple average (by computing the centroidal position)
    if 'centroids' in args.keys() and args['centroids'] > 1:
        while not data[0].shape[0] % args['centroids'] == 0:
            data[0] = data[0][1:, :]
        assert data[0].shape[0] % args['centroids'] == 0, 'Dimensions do not match'

        for i in range(len(data)):
            centroidal_coord = []
            for bidx in range(0, data[i].shape[0], args['centroids']):
                centroidal_coord.append(np.nanmean(
                    data[i][bidx:bidx + args['centroids'], :], axis=0))
            data[i] = np.array(centroidal_coord)

    # invert the Y axis if the user want to (opencv counts 0, 0 from the top left of an image frame)
    for i in range(len(data)):
        if args['invertY']:
            resY = args['resY']
            for n in range(data[i].shape[1] // 2):
                data[i][:, n * 2 + 1] = resY - data[i][:, n * 2 + 1]
        else:
            data[i] = data[i][:min_rows, :]

    # pixel to meter convertion
    for i in range(len(data)):
        # this step should roughly convert pixels to meters
        print('experiment_' + str(i))
        scaled_data = data[i] * args['scale']
        data[i] = filter_func(scaled_data, args)

    # compute setup limits
    info = ExperimentInfo(data)

    # center the data around (0, 0)
    if 'center' in args.keys() and args['center']:
        data, info = Center(data, info).get()

    # normlize data to get them in [-1, 1]
    if 'normalize' in args.keys() and args['normalize']:
        data, info = Normalize(data, info).get()

    return data, info


def last_known(data, args={}):
    """
    :brief: the function will fill in the missing values by replacing them with the last known valid one

    :param data: np.array matrix with missing values that need to be filled in
    :param args: dict, optional extra arguments provided to the function
    :return: np.array matrix without missing values
    """
    filtered_data = []
    for i in range(data.shape[0]):
        row = data[i]
        if np.isnan(row).any():
            idcs = np.where(np.isnan(row) == True)
            if len(filtered_data) < 1:
                continue
            else:
                for idx in idcs[0]:
                    row[idx] = filtered_data[-1][idx]
        filtered_data.append(row)
    return np.array(filtered_data)


def nan_helper(y):
    """
    :param y: np.array of values
    :return: tuple(np.array, lambda) of the nan value indices and a lambda value that applies a transformation on those
    """
    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate(data, args={}):
    """
    :brief: the function will replace missing values by interpolating neighbouring valid ones

    :param data: np.array matrix with missing values that need to be filled in
    :param args: dict, optional extra arguments provided to the function
    :return: np.array matrix without missing values
    """
    for col in range(data.shape[1]):
        nans, x = nan_helper(data[:, col])
        data[nans, col] = np.interp(x(nans), x(~nans), data[~nans, col])
    return data


def skip_zero_movement(data, args={}):
    """
    :brief: the function will remove instances of the trajectories where the individual(s) are not moving faster than
            a set threshold

    :param data: np.array
    :param args: dict, optional extra arguments provided to the function
    :return: np.array
    """
    data = interpolate(data, args)
    if data.shape[1] > 2:
        raise Exception(
            'This filter function should not be used for pair (or more) fish')
    reference = data

    while True:
        zero_movement = 0
        filtered_data = []
        last_row = reference[0, :]
        for i in range(1, reference.shape[0]):
            distance = np.linalg.norm(last_row - reference[i, :])
            last_row = reference[i, :]
            if distance < args['distance_threshold']:
                zero_movement += 1
                continue
            filtered_data.append(reference[i, :])
        reference = np.array(filtered_data)
        if zero_movement == 0:
            break
    filtered_data = reference
    if 'verbose' in args.keys() and args['verbose']:
        print('Lines skipped ' +
              str(data.shape[0] - filtered_data.shape[0]) + ' out of ' + str(data.shape[0]))
    return filtered_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess fish trajectories')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--filename', '-f', type=str,
                        help='Position file name',
                        required=True)
    parser.add_argument('--fps', type=int,
                        help='Camera framerate',
                        required=True)
    parser.add_argument('--centroids', '-c', type=int,
                        help='Frames to use in order to compute the centroidal positions',
                        required=True)
    parser.add_argument('--has-probs', action='store_true',
                        help='Check this flag if the position file contains idTracker positions',
                        default=True)
    parser.add_argument('--toulouse', action='store_true',
                        help='Check this flag if the position file contains the toulouse files',
                        default=False)
    args = parser.parse_args()

    timestep = args.centroids / args.fps

    if not args.toulouse:
        data, files = load(args.path, args.filename, True)
        data, info = preprocess(data,
                                # last_known,
                                # skip_zero_movement,
                                interpolate,
                                args={
                                    'invertY': True,
                                    'resY': 1500,
                                    'scale': 1.12 / 1500,
                                    'initial_keep': 104400,
                                    'centroids': args.centroids,
                                    'distance_threshold': 0.005 * timestep,
                                    'center': True,
                                    'normalize': True,
                                    'verbose': True,
                                    'timestep': timestep
                                })
        info.printInfo()

        velocities = Velocities(data, timestep).get()
        accelerations = Accelerations(velocities, timestep).get()

        archive = Archive({'debug': True})
        for i in range(len(data)):
            f = files[i]
            exp_num = w2n.word_to_num(os.path.basename(
                str(Path(f).parents[0])).split('_')[-1])
            archive.save(data[i], 'exp_' + str(exp_num) +
                         '_processed_positions.dat')
            archive.save(velocities[i], 'exp_' +
                         str(exp_num) + '_processed_velocities.dat')
            archive.save(accelerations[i], 'exp_' +
                         str(exp_num) + '_processed_accelerations.dat')

        with open(archive.path().joinpath('file_order.txt'), 'w') as f:
            for order, exp in enumerate(files):
                f.write(str(order) + ' ' + exp + '\n')
    else:
        data, files = load(args.path, args.filename, False)
        data, info = preprocess(data,
                                last_known,
                                # skip_zero_movement,
                                # interpolate,
                                args={
                                    'invertY': True,
                                    'resY': 1080,
                                    'scale': 0.00058,
                                    # 'initial_keep': 104400,
                                    'centroids': args.centroids,
                                    'distance_threshold': 0.005 * timestep,
                                    'center': True,
                                    'normalize': True,
                                    'verbose': True,
                                    'timestep': timestep
                                })
        info.printInfo()

        velocities = Velocities(data, timestep).get()
        accelerations = Accelerations(velocities, timestep).get()

        archive = Archive({'debug': True})
        for i in range(len(data)):
            f = files[i]
            archive.save(data[i], 'exp_' + str(i) +
                         '_processed_positions.dat')
            archive.save(velocities[i], 'exp_' +
                         str(i) + '_processed_velocities.dat')
            archive.save(accelerations[i], 'exp_' +
                         str(i) + '_processed_accelerations.dat')

        with open(archive.path().joinpath('file_order.txt'), 'w') as f:
            for order, exp in enumerate(files):
                f.write(str(order) + ' ' + exp + '\n')
