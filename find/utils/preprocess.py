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
from copy import copy, deepcopy

from find.utils.features import Velocities
from find.utils.utils import ExperimentInfo, Center, Normalize


class Archive:
    """Serialization class for the fish experiments."""

    def __init__(self, args={}):
        """
        :param args: dict, optional of generic arguments for the class
        """
        if args.out_dir:
            self._experiment_path = args.out_dir
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


def load(exp_path, fname, has_probs=True, args=None):
    """
    :param exp_path: str path to the experiment folder where the data we want to load are stored
    :param fname: str the name of the files we want to load
    :return: tuple(list(np.array), list) of the matrices and corresponding file names
    """
    search_path = exp_path + '/**/'
    files = glob.glob(search_path + fname)
    data = []

    for f in files:
        print('Loading {}'.format(f))
        matrix = np.loadtxt(f, skiprows=1)
        if args is not None and args.bobi:
            time = matrix[:, 0]
            matrix = matrix[:, 1:]
            mhat = np.empty((matrix.shape[0], 0))
            for i in range(matrix.shape[1] // 5):
                cut = matrix[:, (i*5):(i*5 + 2)]
                mhat = np.hstack([mhat, cut])
            matrix = mhat

            if args.skip_closely_stamped:
                idcs = [0]
                for i in range(1, time.shape[0]):
                    if time[i] - time[idcs[-1]] >= (1 / args.fps) * 0.97:
                        idcs.append(i)

                matrix = matrix[idcs, :]
                print('Skipped {} frames that were stamped in less than {:.3f} s between them'.format(
                    len(time) - matrix.shape[0], (1 / args.fps) * 0.97))

            # this is originally values that were nan but saves as a big number to avoid problems with some coding languages when loading
            matrix[np.where(matrix > 5000)] = np.nan
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

    # invert the Y axis if the user want to (opencv counts 0, 0 from the top left of an image frame)
    for i in range(len(data)):
        if args['invertY']:
            resY = args['resY']
            for n in range(data[i].shape[1] // 2):
                data[i][:, n * 2 + 1] = resY - data[i][:, n * 2 + 1]

    for i in tqdm.tqdm(range(len(data)), desc='Interpolating'):
        data[i] = interpolate(data[i], args)

    info = ExperimentInfo(data)

    if 'diameter_allowed_error' in args.keys():
        diameters = []
        for i in range(len(data)):
            xminh = info.minXY(i)[0]
            xmaxh = info.maxXY(i)[0]
            yminh = info.minXY(i)[1]
            ymaxh = info.maxXY(i)[1]
            maxdh = max([xmaxh-xminh, ymaxh-yminh])
            diameters.append(maxdh)
        diameter_thres = np.median(diameters) * args['diameter_allowed_error']
        diameter_diff = [np.abs(x - np.median(x)) for x in diameters]

    # pixel to meter convertion
    for i in range(len(data)):
        # this step should roughly convert pixels to meters
        print('experiment_' + str(i))
        if args['scale'] < 0:
            assert ('radius' in args.keys(
            )), 'Automatic scaling factor computation requires knowledge of the radius'

            xmin = info.globalMinXY()[0]
            xmax = info.globalMaxXY()[0]
            ymin = info.globalMinXY()[1]
            ymax = info.globalMaxXY()[1]
            maxd = max([xmax-xmin, ymax-ymin])

            if 'use_global_min_max' in args.keys() and not args['use_global_min_max']:
                xminh = info.minXY(i)[0]
                xmaxh = info.maxXY(i)[0]
                yminh = info.minXY(i)[1]
                ymaxh = info.maxXY(i)[1]
                maxdh = max([xmaxh-xminh, ymaxh-yminh])

                if 'diameter_allowed_error' in args.keys() and diameter_diff[i] > diameter_thres:
                    maxd = maxdh

            a = 2 * args['radius'] / maxd
            data[i] = data[i] * a
        else:
            data[i] = data[i] * args['scale']

    info = ExperimentInfo(data)
    info.printInfo()

    # here we attempt to address the tracking misclassification of the trajectories causing sudden jumps
    for i in tqdm.tqdm(range(len(data)), desc='Disentangling trajectories'):
        data[i] = correct_trajectories(data[i], args)

    # this is the main filtering function selected by the user. Although even the jumping can be included in this section
    # we opted to separate the two so as to allow more freedom for people that want to implement custom functions
    idcs_remove = []
    for i in tqdm.tqdm(range(len(data)), desc='Filtering'):
        data[i] = filter_func(data[i], args)
        if data[i].shape[0] < (args['min_seq_len'] / (args['timestep'] / args['centroids'])):
            idcs_remove.append(i)

    idcs_removed = 0
    for i, idx in tqdm.tqdm(enumerate(idcs_remove), desc='Removing small files after filtering'):
        del data[idx - idcs_removed]
        del files[idx - idcs_removed]
        idcs_removed += 1

    # remove jumping instances, that is, if an individual travels an unusually great distance
    # which could be an indication that the tracking was momentarily confused
    idx_correction = {}
    if 'jump_threshold' in args.keys():
        odata = deepcopy(data)
        oinfo = ExperimentInfo(odata)
        data, files, idx_correction = correct_jumping(data, files, args)

        idcs_remove = []
        for i in range(len(data)):
            # skip files that are less than args['min_seq_len'] seconds long
            if data[i].shape[0] < (args['min_seq_len'] / (args['timestep'] / args['centroids'])):
                idcs_remove.append(i)

        idcs_removed = 0
        for idx in tqdm.tqdm(idcs_remove, desc='Removing small files'):
            del data[idx - idcs_removed]
            del files[idx - idcs_removed]
            k = list(idx_correction.keys())[idx-idcs_removed]
            del idx_correction[k]
            idcs_removed += 1

    # filtering the data with a simple average (by computing the centroidal position)
    if 'centroids' in args.keys() and args['centroids'] > 1:
        while not data[0].shape[0] % args['centroids'] == 0:
            for i in range(len(data)):
                data[i] = data[i][1:, :]
        assert data[0].shape[0] % args['centroids'] == 0, 'Dimensions do not match'

        for i in range(len(data)):
            centroidal_coord = []
            for bidx in range(0, data[i].shape[0], args['centroids']):
                centroidal_coord.append(np.nanmean(
                    data[i][bidx:bidx + args['centroids'], :], axis=0))
            data[i] = np.array(centroidal_coord)

    # compute setup limits
    info = ExperimentInfo(data)

    if 'jump_threshold' in args.keys():
        for k, idx in idx_correction.items():
            i = list(idx_correction.keys()).index(k)
            minXY = oinfo.minXY(idx)
            maxXY = oinfo.maxXY(idx)
            info.setMinXY(minXY, i)
            info.setMaxXY(maxXY, i)

    # center the data around (0, 0)
    if 'center' in args.keys() and args['center']:
        data, info = Center(data, info, args).get()
        if 'jump_threshold' in args.keys():
            odata, oinfo = Center(odata, oinfo).get()
            for i, (k, idx) in enumerate(idx_correction.items()):
                minXY = oinfo.minXY(idx)
                maxXY = oinfo.maxXY(idx)
                info.setMinXY(minXY, i)
                info.setMaxXY(maxXY, i)

    # normlize data to get them in [-1, 1]
    if 'normalize' in args.keys() and args['normalize']:
        data, info = Normalize(data, info, args).get()
        if 'jump_threshold' in args.keys():
            odata, oinfo = Normalize(odata, oinfo).get()
            for i, (k, idx) in enumerate(idx_correction.items()):
                minXY = info.minXY(idx)
                maxXY = info.maxXY(idx)
                info.setMinXY(minXY, i)
                info.setMaxXY(maxXY, i)
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


def correct_trajectories(data, args={}):
    for i in range(1, data.shape[0]):
        for ind in range(data.shape[1] // 2):
            ref = data[i, (ind * 2): (ind * 2 + 2)]
            distances = [np.linalg.norm(
                ref - data[i-1, (x * 2): (x * 2 + 2)]) for x in range(data.shape[1] // 2)]
            idx_min = np.argmin(distances)
            if distances[idx_min] < np.linalg.norm(data[i, (idx_min * 2):(idx_min * 2 + 2)] - data[i-1, (idx_min * 2):(idx_min * 2 + 2)]):
                tmp = data[i, (ind * 2): (ind * 2 + 2)]
                data[i, (ind * 2): (ind * 2 + 2)] = data[i,
                                                         (idx_min * 2): (idx_min * 2 + 2)]
                data[i, (idx_min * 2): (idx_min * 2 + 2)] = tmp
    return data


def correct_jumping(data, files, args={'jump_threshold': 0.08}):
    new_data = deepcopy(data)

    # bootstrap the inverse index table
    idf_idx_track = {}
    for i in range(len(data)):
        idf_idx_track[i] = i

    it = 0
    while it < len(new_data):
        data_it = new_data[it]
        stop_it = -1

        for i in range(1, data_it.shape[0]):
            for ind in range(data_it.shape[1] // 2):
                ref = data_it[i, (ind * 2): (ind * 2 + 2)]
                ref_prev = data_it[i-1, (ind * 2): (ind * 2 + 2)]
                distance = np.linalg.norm(ref - ref_prev)
                if distance > args['jump_threshold']:
                    stop_it = i
                    break

            if stop_it > 0:
                break

        if stop_it >= 0:
            new_data[it] = data_it[:stop_it, :]
            new_data.append(data_it[stop_it:, :])
            if 'split' not in files[it]:
                files.append(files[it].replace('.dat', '_split.dat'))
            else:
                files.append(files[it])

            idf_idx_track[len(files)-1] = idf_idx_track[it]

            print('Splitting file ' + files[it] +
                  ' at timestep ' + str(stop_it))
        else:
            new_data[it] = data_it

        it += 1
    return new_data, files, idf_idx_track


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
    idcs_remove = []

    window = args['window']
    hwindow = window // 2
    idcs_remove = []
    for ind in range(data.shape[1] // 2):
        reference = data[:, (ind * 2): (ind * 2 + 2)]
        for i in tqdm.tqdm(range(1, reference.shape[0]), desc='Checking movement in window for individual ' + str(ind)):
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

    if data.shape[0] - filtered_data.shape[0] > 0:
        return skip_zero_movement(filtered_data, args)
    else:
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
        r = np.sqrt((data[:, i * 2] - center[0]) ** 2 +
                    (data[:, i * 2 + 1] - center[1]) ** 2)
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
            row = fill_circle(
                i, data[:, (idx * 2): (idx * 2 + 2)], output_matrix, center)
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
    parser.add_argument('--out_dir', type=str,
                        help='Explicit output directory path',
                        default='',
                        required=False)
    parser.add_argument('--fps', type=int,
                        help='Camera framerate',
                        required=True)
    parser.add_argument('--centroids', '-c', type=int,
                        help='Frames to use in order to compute the centroidal positions',
                        required=True)
    parser.add_argument('--has_probs', action='store_true',
                        help='Check this flag if the position file contains idTracker positions',
                        default=True)
    parser.add_argument('--toulouse', action='store_true',
                        help='Check this flag if the position file contains the toulouse files',
                        default=False)
    parser.add_argument('--bobi', action='store_true',
                        help='Check this flag if the position file contains the BOBI files',
                        default=False)
    parser.add_argument('--skip_closely_stamped', action='store_true',
                        help='(BOBI only) skip samples that are very closely stamped (in time)',
                        default=True)
    parser.add_argument('--plos', action='store_true',
                        help='Check this flag if the position file contains the plos files',
                        default=False)
    parser.add_argument('--bl', type=float,
                        help='Body length',
                        default=0.035,
                        required=False)
    parser.add_argument('--radius', type=float,
                        help='Radius for circular setups',
                        default=0.25,
                        required=False)
    parser.add_argument('--min_seq_len', type=float,
                        help='Minimum sequence length in seconds to keep when filtering',
                        default=0.6,
                        required=False)
    parser.add_argument('--robot',
                        action='store_true',
                        help='If this was robot experiments, then look for the robot index files',
                        default=False,
                        required=False)
    args = parser.parse_args()

    timestep = args.centroids / args.fps
    archive = Archive(args)

    if args.toulouse:
        data, files = load(args.path, args.filename, False, args)
        data, info, files = preprocess(data, files,
                                       #    last_known,
                                       skip_zero_movement,
                                       #    interpolate,
                                       args={
                                           'use_global_min_max': False,
                                           'diameter_allowed_error': 0.15,

                                           'invertY': False,
                                           'resY': 1080,
                                           'scale': -1,  # automatic scale detection
                                           'radius': args.radius,
                                           'centroids': args.centroids,
                                           'distance_threshold': args.bl * 1.2,
                                           'jump_threshold': args.bl * 1.5,
                                           'window': 30,

                                           'is_circle': True,
                                           'center': True,
                                           'normalize': True,
                                           'verbose': True,
                                           'timestep': timestep,

                                           'min_seq_len': args.min_seq_len,

                                       })
        info.printInfo()

        velocities = Velocities(data, timestep).get()

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
        data, files = load(args.path, args.filename, True, args)
        data, info, files = preprocess(data, files,
                                       # last_known,
                                       # skip_zero_movement,
                                       # interpolate,
                                       cspace,
                                       args={
                                           'invertY': True,
                                           'resY': 1024,
                                           'scale': 1.11 / 1024,
                                           'centroids': args.centroids,
                                           'distance_threshold': 0.00875,
                                           'center': True,
                                           'normalize': True,
                                           'verbose': True,
                                           'timestep': timestep,

                                           'min_seq_len': args.min_seq_len,
                                       })
        info.printInfo()

        velocities = Velocities(data, timestep).get()

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
    elif args.bobi:
        data, files = load(args.path, args.filename, False, args)
        robot_idcs = {}
        for f in files:
            exp_num = w2n.word_to_num(os.path.basename(
                str(Path(f).parents[0])).split('_')[-1])

            if args.robot:
                if not os.path.exists(f.replace('.txt', '_ridx.txt')):
                    assert False, 'Robot index file missing for: {}'.format(f)
                idx = np.array(
                    [np.loadtxt(f.replace('.txt', '_ridx.txt')).astype(int)])
                robot_idcs[exp_num] = idx
            else:
                robot_idcs[exp_num] = np.array([-1])

        if args.fps == 30:
            window = 36
        else:
            window = 30
        data, info, files = preprocess(data, files,
                                       # last_known,
                                       skip_zero_movement,
                                       #    interpolate,
                                       # cspace,
                                       args={
                                           'use_global_min_max': False,
                                           'diameter_allowed_error': 0.15,

                                           'invertY': False,
                                           #    'resY': 1500,
                                           'scale': -1,  # automatic scale detection
                                           #    'scale': 1.12 / 1500,
                                           'radius': args.radius,

                                           'centroids': args.centroids,
                                           'distance_threshold': args.bl * 1.2,
                                           'jump_threshold': args.bl * 1.5,
                                           'window': window,

                                           'is_circle': True,
                                           'center': True,
                                           'normalize': True,
                                           'verbose': True,
                                           'timestep': timestep,

                                           'min_seq_len': args.min_seq_len,
                                       })

        velocities = Velocities(data, timestep).get()
        for i in range(len(data)):
            f = files[i]
            exp_num = w2n.word_to_num(os.path.basename(
                str(Path(f).parents[0])).split('_')[-1])
            archive.save(
                data[i], 'exp_{}-{}_processed_positions.dat'.format(exp_num, i))
            archive.save(
                velocities[i], 'exp_{}-{}_processed_velocities.dat'.format(exp_num, i))
            archive.save(robot_idcs[exp_num].astype(
                int), 'exp_{}-{}_processed_positions_ridx.dat'.format(exp_num, i))

        with open(archive.path().joinpath('file_order.txt'), 'w') as f:
            for order, exp in enumerate(files):
                f.write(str(order) + ' ' + exp + '\n')
    else:
        data, files = load(args.path, args.filename, False, args)
        robot_idcs = {}
        for f in files:
            exp_num = w2n.word_to_num(os.path.basename(
                str(Path(f).parents[0])).split('_')[-1])

            if args.robot:
                if not os.path.exists(f.replace('.txt', '_ridx.txt')):
                    assert False, 'Robot index file missing for: {}'.format(f)
                idx = np.array(
                    [np.loadtxt(f.replace('.txt', '_ridx.txt')).astype(int)])
                robot_idcs[exp_num] = idx
            else:
                robot_idcs[exp_num] = np.array([-1])

        if args.fps == 30:
            window = 36
        else:
            window = 30

        data, info, files = preprocess(data, files,
                                       # last_known,
                                       skip_zero_movement,
                                       #    interpolate,
                                       # cspace,
                                       args={
                                           'use_global_min_max': False,
                                           'diameter_allowed_error': 0.15,

                                           'invertY': False,
                                           #    'resY': 1500,
                                           'scale': -1,  # automatic scale detection
                                           #    'scale': 1.12 / 1500,
                                           'radius': args.radius,

                                           'centroids': args.centroids,
                                           'distance_threshold': args.bl * 1.2,
                                           'jump_threshold': args.bl * 1.5,
                                           'window': window,

                                           'is_circle': True,
                                           'center': True,
                                           'normalize': True,
                                           'verbose': True,
                                           'timestep': timestep,

                                           'min_seq_len': args.min_seq_len,
                                       })
        info.printInfo()

        velocities = Velocities(data, timestep).get()

        for i in range(len(data)):
            f = files[i]
            exp_num = w2n.word_to_num(os.path.basename(
                str(Path(f).parents[0])).split('_')[-1])
            archive.save(
                data[i], 'exp_{}-{}_processed_positions.dat'.format(exp_num, i))
            archive.save(
                velocities[i], 'exp_{}-{}_processed_velocities.dat'.format(exp_num, i))
            archive.save(robot_idcs[exp_num].astype(
                int), 'exp_{}-{}_processed_positions_ridx.dat'.format(exp_num, i))

        with open(archive.path().joinpath('file_order.txt'), 'w') as f:
            for order, exp in enumerate(files):
                f.write(str(order) + ' ' + exp + '\n')
