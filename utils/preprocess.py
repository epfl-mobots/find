#!/usr/bin/env python
import tqdm
import argparse
import datetime
import glob
import numpy as np
import os
import socket  # for get hostname
from pathlib import Path
from word2number import w2n
from pprint import pprint

from features import Velocities
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


def preprocess(data, files, filter_func, args={'scale': 1.0}):
    """
    :param data: list(np.array) of position data for different fish individuals or experiments
    :param files: list(str) of position files
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
        if args['scale'] < 0:
            assert ('radius' in args.keys()), 'Automatic scaling factor computation requires knowledge of the radius'
            info = ExperimentInfo(data)
            info.printInfo()
            xmin = info.globalMinXY()[0]
            xmax = info.globalMaxXY()[0]

            a = 2 * args['radius'] / (xmax-xmin)
            print('Scaling factor: ', str(a))
            data[i] = data[i] * a
        else:
            data[i] = data[i] * args['scale']

    if 'jump_threshold' in args.keys():
        for i in range(len(data)):
            data[i] = interpolate(data[i], args) 
        data, files = correct_jumping(data, files, args)     
 
        idcs_remove = []
        for i in range(len(data)):
            if data[i].shape[0] < (2.0 / (args['timestep'] / args['centroids'])): # skip files that are less than 0.6 seconds long
                idcs_remove.append(i)    

        idcs_removed = 0
        for idx in idcs_remove:
            del data[idx - idcs_removed]
            del files[idx - idcs_removed]
            idcs_removed += 1

    idcs_remove = []
    for i in tqdm.tqdm(range(len(data))):
        data[i] = filter_func(data[i], args)
        if data[i].shape[0] < (2.0 / (args['timestep'] / args['centroids'])):
            idcs_remove.append(i)    

    idcs_removed = 0
    for idx in idcs_remove:
        del data[idx - idcs_removed]
        del files[idx - idcs_removed]
        idcs_removed += 1

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

    # compute setup limits
    info = ExperimentInfo(data)

    # center the data around (0, 0)
    if 'center' in args.keys() and args['center']:
        data, info = Center(data, info).get()

    # normlize data to get them in [-1, 1]
    if 'normalize' in args.keys() and args['normalize']:
        data, info = Normalize(data, info).get()

    return data, info, files


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


def correct_jumping(data, files, args={'jump_threshold': 0.08}):
    new_data = data

    it = 0
    while it < len(new_data):
        data_it = new_data[it]
        stop_it = -1
        for i in range(1, data_it.shape[0]):
            for ind in range(data_it.shape[1] // 2):
                ref = data_it[i, (ind * 2) : (ind * 2 + 2)]    
                ref_prev = data_it[i-1, (ind * 2) : (ind * 2 + 2)]    
                distance = np.linalg.norm(ref - ref_prev)
                if distance > args['jump_threshold']:
                    distances = [np.linalg.norm(ref - data_it[i-1, (x * 2) : (x * 2 + 2)]) for x in range(data_it.shape[1] // 2)]
                    idx_min = np.argmin(distances)
 
                    if distances[idx_min] > args['jump_threshold']:
                        stop_it = i
                        break
                    else:
                        data_it[i, (ind * 2) : (ind * 2 + 2)] = data_it[i, (idx_min * 2) : (idx_min * 2 + 2)]

            if stop_it > 0:
                break
        if not stop_it > 0:
            new_data[it] = data_it
        else:
            new_data[it] = data_it[:stop_it, :]
            new_data.append(data_it[stop_it:, :])
            if 'split' not in files[it]:
                files.append(files[it].replace('raw', 'split_raw'))
            else:
                files.append(files[it])
            print('Splitting file ' + files[it] + ' at timestep ' + str(stop_it))
        it += 1
    return new_data, files


def skip_zero_movement(data, args={'window': 30}):
    """
    :brief: the function will remove instances of the trajectories where the individual(s) are not moving faster than
            a set threshold

    :param data: np.array
    :param args: dict, optional extra arguments provided to the function
    :return: np.array
    """

    window = args['window']
    hwindow = window // 2
    data = interpolate(data, args)
    idcs_remove = []
    for ind in range(data.shape[1] // 2):
        reference = data[:, (ind * 2) : (ind * 2 + 2)]    
        for i in range(reference.shape[0]):
            lb = max([0, i - hwindow])
            ub = min([i + hwindow, reference.shape[0]]) 

            last_row = reference[i-1, :]
            distance_covered = 0

            for w in range(lb, ub):
                distance_covered += np.linalg.norm(last_row - reference[w, :])
                last_row = reference[w, :]

            if distance_covered < args['distance_threshold']:
                idcs_remove += [i]

    idcs_remove = list(set(idcs_remove))
    if len(idcs_remove) > 0:
        filtered_data = np.delete(data, idcs_remove, axis=0)
    else:
        filtered_data = data

    if 'verbose' in args.keys() and args['verbose']:
        print('Lines skipped ' +
            str(data.shape[0] - filtered_data.shape[0]) + ' out of ' + str(data.shape[0]))
    return filtered_data


def cspace(data, args={}):
    """
    :brief: Check if given row index corresponds to an invalid position and if does fill it with a value corresponding to a circular trajectory
    :param row_idx: int, index of the position to check
    :param xy: numpy.array, matrix of x, y positions
    :param output_matrix: list, valid points found so far
    :param args: dict, additional arguments for the fitting method
    :return: list, x, y replacement x, y coordinates along a circular path
    """

    info = ExperimentInfo([interpolate(data, args)])
    if not 'radius' in args.keys():
        radius = (0.29, 0.19)
    else:
        radius = args['radius']
    center = info.center()

    for i in range(data.shape[1] // 2):
        r = np.sqrt((data[:, i * 2] - center[0]) ** 2 + (data[:, i * 2 + 1] - center[1]) ** 2)
        idcs = np.where(r < radius[1])
        data[idcs, i * 2] = np.nan
        data[idcs, i * 2 + 1] = np.nan

    def angle_to_pipi(dif):
        while True:
            if dif < -np.pi:
                dif += 2. * np.pi
            if dif > np.pi:
                dif -= 2. * np.pi
            if (np.abs(dif) <= np.pi):
                break
        return dif

    def find_next(xy, row_idx):
        """
        :brief: Given a matrix of xy positions and the current timestep, find the next valid (non nan) point
        :param xy: numpy.array, matrix of x, y positions (m x 2 where m the number of timsteps)
        :param row_idx: int, current timestep
        :return: tuple, the next known point accompanied by its index in the position matrix
        """

        next_known = (-1, -1)
        for i in range(row_idx + 1, np.shape(xy)[0]):
            if not np.isnan(xy[i]).any():
                next_known = (xy[i], i - row_idx)
                break
        return next_known


    def fit_circle(pt1, pt2, center):
        """
        :brief: Fit a circle between two points and a given center
        :param pt1: tuple, x, y positions for the first point
        :param pt2: tuple, x, y positions for the second point
        :param center: tuple, desired center for the fitted circle
        :return: tuple(float, tuple(float, float, float)), r is the radius of the fitted circle,
                                                                                theta the angle between the new points,
                                                                                theta1, theta2 the angle of pt1 and pt2
                                                                                starting from zero, respectively
        """

        r1 = np.sqrt((pt1[0] - center[0]) ** 2 + (pt1[1] - center[1]) ** 2)
        r2 = np.sqrt((pt2[0] - center[0]) ** 2 + (pt2[1] - center[1]) ** 2)
        r = (r1 + r2) / 2.0

        theta1 = np.arctan2(pt1[1], pt1[0])
        theta1 = (theta1 + np.pi) % (2 * np.pi)
        theta2 = np.arctan2(pt2[1], pt2[0])
        theta2 = (theta2 + np.pi) % (2 * np.pi)
        theta = angle_to_pipi(theta1 - theta2)
        return r, (theta, theta1, theta2)


    def fill_between_circular(last_known, next_known, center):
        """
        :brief: Fill a circular trajectory with mising values given the first and next valid positions
        :param last_known: list, the last known valid x, y position
        :param next_known: list, the next known point and the number of missing values between this and the last known
        :return: list, x, y position that was estimated according to the given points
        """

        r, (theta, theta1, _) = fit_circle(
            np.array(last_known), np.array(next_known[0]), center)
        sgn = np.sign(theta)
        phi = (np.abs(theta) / (next_known[1] + 1))  # step angle
        estimated = [r * np.cos(theta1 - sgn * phi) + center[0],
                    r * np.sin(theta1 - sgn * phi) + center[1]]
        return estimated


    def fill_forward_circular(second_last_known, last_known, args):
        """
        :brief: Given the two last known positions and the center of a circular setup,
                attempt to find the next position of a missing trajectory
        :param second_last_known: list, the second to last known position
        :param last_known: list, the last known valid x, y position
        :return: tuple, the next known point accompanied by its index in the position matrix
        """

        r, (theta, _, theta2) = fit_circle(
            np.array(second_last_known), np.array(last_known), center)
        sgn = np.sign(theta)
        phi = np.abs(theta)  # step angle
        return [r * np.cos(theta2 - sgn * phi) + center[0],
                r * np.sin(theta2 - sgn * phi) + center[1]]

    def fill_circle(row_idx, xy, output_matrix, args):
        """
        :brief: Main logic to fill missing trajectory values with circular ones
        :param row_idx: int, current row of interest index
        :param xy: np.array, original trajectory matrix
        :param output_matrix: np.array, containing the corrected positions so far
        :param args: dict, additional arguments for the algorithm
        :return: np.array, valid row with circular tajectory
        """
        row = xy[row_idx]

        if np.isnan(row).any():
            next_known = find_next(xy, row_idx)

            if len(output_matrix) < 1:  # continue trajectory according to next two known positions
                second_next = find_next(xy, row_idx + next_known[1] + 1)
                if second_next[1] > 1:
                    second_next = (fill_between_circular(
                        next_known[0], second_next, center), -1)
                return fill_forward_circular(second_next[0], next_known[0], center)
            else:
                last_known = output_matrix[-1]
                if next_known[1] > 0:
                    return fill_between_circular(last_known, next_known, center)
                else:  # continue trajectory according to last two known positions
                    return fill_forward_circular(output_matrix[-2], last_known, center)
        else:
            return row


    for idx in range(data.shape[1] // 2):
        output_matrix = []
        for i in range(data.shape[0]):
            row = fill_circle(i, data[:, (idx * 2) : (idx * 2 + 2)], output_matrix, center)
            if len(row) == 0:
                continue
            output_matrix.append(row)
        corrected_matrix = np.array(output_matrix)
        data[:, idx * 2] = corrected_matrix[:, 0]
        data[:, idx * 2 + 1] = corrected_matrix[:, 1]
    return data


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
    parser.add_argument('--plos', action='store_true',
                        help='Check this flag if the position file contains the plos files',
                        default=False)
    parser.add_argument('--bl', type=float,
                        help='Body length',
                        default=0.035,
                        required=False)
    args = parser.parse_args()

    timestep = args.centroids / args.fps

    if args.toulouse:
        data, files = load(args.path, args.filename, False)
        data, info, files = preprocess(data, files,
                                # last_known,
                                skip_zero_movement,
                                # interpolate,
                                args={
                                    'invertY': True,
                                    'resY': 1080,
                                    'scale': -1, # automatic scale detection
                                    'radius': 0.25,
                                    'centroids': args.centroids,
                                    'distance_threshold': args.bl * 0.75, 
                                    'jump_threshold': args.bl * 1.2,
                                    'window': 30, 

                                    'center': True,
                                    'normalize': True,
                                    'verbose': True,
                                    'timestep': timestep
                                })
        info.printInfo()

        velocities = Velocities(data, timestep).get()

        archive = Archive({'debug': True})
        for i in range(len(data)):
            f = files[i]
            archive.save(data[i], 'exp_' + str(i) +
                         '_processed_positions.dat')
            archive.save(velocities[i], 'exp_' +
                         str(i) + '_processed_velocities.dat')

        with open(archive.path().joinpath('file_order.txt'), 'w') as f:
            for order, exp in enumerate(files):
                f.write(str(order) + ' ' + exp + '\n')
    elif args.plos:
        data, files = load(args.path, args.filename, True)
        data, info, files = preprocess(data, files,
                                # last_known,
                                # skip_zero_movement,
                                # interpolate,
                                cspace,
                                args={
                                    'invertY': True,
                                    'resY': 1024,
                                    'scale': 1.11 / 1024 ,
                                    'centroids': args.centroids,
                                    'distance_threshold': 0.00875,
                                    'center': True,
                                    'normalize': True,
                                    'verbose': True,
                                    'timestep': timestep
                                })
        info.printInfo()

        velocities = Velocities(data, timestep).get()

        archive = Archive({'debug': True})
        for i in range(len(data)):
            f = files[i]
            exp_num = w2n.word_to_num(os.path.basename(
                str(Path(f).parents[0])).split('_')[-1])
            archive.save(data[i], 'exp_' + str(exp_num) +
                         '_processed_positions.dat')
            archive.save(velocities[i], 'exp_' +
                         str(exp_num) + '_processed_velocities.dat')

        with open(archive.path().joinpath('file_order.txt'), 'w') as f:
            for order, exp in enumerate(files):
                f.write(str(order) + ' ' + exp + '\n')
    else:
        data, files = load(args.path, args.filename, True)
        data, info, files = preprocess(data, files,
                                # last_known,
                                # skip_zero_movement,
                                interpolate,
                                # cspace,
                                args={
                                    'invertY': True,
                                    'resY': 1500,
                                    'scale': 1.12 / 1500,
                                    'initial_keep': 104400,
                                    'centroids': args.centroids,
                                    'distance_threshold': 0.00875,
                                    'center': True,
                                    'normalize': True,
                                    'verbose': True,
                                    'timestep': timestep
                                })
        info.printInfo()

        velocities = Velocities(data, timestep).get()

        archive = Archive({'debug': True})
        for i in range(len(data)):
            f = files[i]
            exp_num = w2n.word_to_num(os.path.basename(
                str(Path(f).parents[0])).split('_')[-1])
            archive.save(data[i], 'exp_' + str(exp_num) +
                         '_processed_positions.dat')
            archive.save(velocities[i], 'exp_' +
                         str(exp_num) + '_processed_velocities.dat')

        with open(archive.path().joinpath('file_order.txt'), 'w') as f:
            for order, exp in enumerate(files):
                f.write(str(order) + ' ' + exp + '\n')

