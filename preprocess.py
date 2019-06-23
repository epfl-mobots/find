#!/usr/bin/env python
import os
import glob
import socket # for get hostname
import argparse
import datetime
import numpy as np
from pathlib import Path
from pprint import pprint
from word2number import w2n


class Archive:
    def __init__(self, args={'debug' : False}):        
        if args['debug']:
            self._experiment_path = 'test'
        else:
            self._hostname = socket.gethostname()
            self._timestamp = datetime.date.today().strftime('%Y_%m_%d') + '-' + datetime.datetime.now().strftime('%H_%M_%S')
            self._experiment_path = self._hostname + '_' + self._timestamp

        if not os.path.exists(self._experiment_path):
            os.makedirs(self._experiment_path)


    def path(self):
        return Path(self._experiment_path)


    def save(self, data, filename):
        if isinstance(data, (np.ndarray, np.generic)):
            np.savetxt(self.path().joinpath(filename), data)
        else:
            assert False, 'Can not store data structures of this type'



class ExperimentInfo:
    def __init__(self, data):
        self._maxXs = [np.max(np.delete(matrix, np.s_[1::2], 1)) for matrix in data]
        self._minXs = [np.min(np.delete(matrix, np.s_[1::2], 1)) for matrix in data]
        self._global_minX = np.min(self._minXs)
        self._global_maxX = np.max(self._maxXs)

        self._maxYs = [np.max(np.delete(matrix, np.s_[0::2], 1)) for matrix in data]
        self._minYs = [np.min(np.delete(matrix, np.s_[0::2], 1)) for matrix in data]
        self._global_minY = np.min(self._minYs)
        self._global_maxY = np.max(self._maxYs)


    def center(self, idx=-1):
        if idx < 0:
            return ((self._global_maxX + self._global_minX) / 2, (self._global_maxY + self._global_minY) / 2)
        else:
            return ((self._maxXs[idx] + self._minXs[idx]) / 2, (self._maxYs[idx] + self._minYs[idx]) / 2)


    def minXY(self, idx):
        return (self._minXs[idx], self._minYs[idx])


    def maxXY(self, idx):
        return (self._maxXs[idx], self._maxYs[idx])


    def print(self):
        print('Center: ' + str(self.center()))
        print('min(X, Y): ' + str(self._global_minX) + ', ' + str(self._global_minY))
        print('max(X, Y): ' + str(self._global_maxX) + ', ' + str(self._global_maxY))


class Velocities:
    def __init__(self, positions, timestep):
        self._velocities = []
        for p in positions:            
            rolled_p = np.roll(p, shift=1, axis=0)
            velocities = (rolled_p - p) / timestep
            mu = np.mean(velocities[:-1, :], axis=0)
            sigma = np.std(velocities[:-1, :], axis=0)

            x_rand = np.random.normal(mu[0], sigma[0], p.shape[1] // 2)
            y_rand = np.random.normal(mu[1], sigma[1], p.shape[1] // 2)
            for i in range(p.shape[1] // 2):
                velocities[-1, i * 2] = x_rand[i]
                velocities[-1, i * 2 + 1] = y_rand[i] 
            self._velocities.append(velocities)       


    def get(self):
        return self._velocities


def load(exp_path, fname):
    files = glob.glob(exp_path + '/**/' + fname)
    data = []
    for f in files:
        matrix = np.loadtxt(f, skiprows=1)
        matrix = np.delete(matrix, np.s_[2::3], 1)
        data.append(matrix)
    return data, files


def preprocess(data, filter_func, args={'scale' : 1.0}):
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
        assert data[0].shape[0] % args['centroids'] == 0, 'Dimensions do not match'
        for i in range(len(data)):
            centroidal_coord = []
            for bidx in range(0, data[i].shape[0], args['centroids']):
                centroidal_coord.append(np.nanmean(data[i][bidx:bidx+args['centroids'], :] , axis=0))
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
        scaled_data = data[i] * args['scale']
        data[i] = filter_func(scaled_data, args)

    # compute setup limits
    info = ExperimentInfo(data) 

    # center the data around (0, 0) 
    if 'center' in args.keys() and args['center']:
        for i, matrix in enumerate(data):
            c = info.center(i)
            for n in range(matrix.shape[1] // 2):
                matrix[:, n * 2] =  matrix[:, n * 2] - c[0] 
                matrix[:, n * 2 + 1] = matrix[:, n * 2 + 1] - c[1]
        info = ExperimentInfo(data) 

    # normlize data to get them in [-1, 1]
    if 'normalize' in args.keys() and args['normalize']:
        for i, matrix in enumerate(data):
            maxXY = info.maxXY(i)
            for n in range(matrix.shape[1] // 2):
                matrix[:, n * 2] /= maxXY[0] 
                matrix[:, n * 2 + 1] /= maxXY[1]
        info = ExperimentInfo(data)         

    return data, info


def last_known(data, args={}):
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


def skip_zero_movement(data, args={}):
    eps = args['eps']
    data = last_known(data, args)
    reference = data
    while(True):
        zero_movement = 0
        filtered_data = []
        last_row = reference[0, :]
        for i in range(1, reference.shape[0]):
            mse = np.linalg.norm(last_row - reference[i, :])
            last_row = reference[i, :]
            if mse <= eps:
                zero_movement += 1
                continue
            filtered_data.append(reference[i, :])
        reference = np.array(filtered_data)
        if zero_movement == 0:
            break
    filtered_data = reference 
    if 'verbose' in args.keys() and args['verbose']:  
        print('Lines skipped ' + str(data.shape[0] - filtered_data.shape[0]) + ' out of ' + str(data.shape[0]))
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
    args = parser.parse_args()

    timestep = args.centroids / args.fps

    data, files = load(args.path, args.filename)
    data, info = preprocess(data, skip_zero_movement, 
        args={
            'invertY' : True,
            'resY' : 1500,
            'scale' : 1.12 / 1500,
            'initial_keep' : 104400,
            'centroids' : 3, 
            'eps': 0.00006,
            'center' : True,
            'normalize' : True,
            'verbose' : False,
        })
    info.print()

    velocities = Velocities(data, timestep).get()

    archive = Archive({'debug' : True})
    for i, f in enumerate(files):
        exp_num = w2n.word_to_num(os.path.basename(Path(f).parents[0]).split('_')[-1])
        archive.save(data[i], 'exp_' + str(exp_num) + '_processed_positions.dat')                 
        archive.save(velocities[i], 'exp_' + str(exp_num) + '_processed_velocities.dat')                 
