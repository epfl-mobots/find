#!/usr/bin/env python
import glob
import argparse
import numpy as np
from pprint import pprint

class ExperimentInfo:
    minX = float('Inf')
    maxX = -float('Inf')
    minY = float('Inf')
    maxY = -float('Inf')

    def center(self):
        return ((self.maxX + self.minX / 2), (self.maxY + self.minY / 2))

def load(exp_path, fname):
    files = glob.glob(exp_path + '/**/' + fname)

    data = []
    for f in files:
        matrix = np.loadtxt(f, skiprows=1)
        matrix = np.delete(matrix, np.s_[2::3], 1)
        data.append(matrix)
    return data

def filter(data, filter_func, args={'scale' : 1.0}):
    # every matrix should have the same number of rows
    if 'initial_keep' in args.keys():
        for i in range(len(data)):
            data[i] = data[i][:args['initial_keep'], :]
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
                centroidal_coord.append(np.mean(data[i][bidx:bidx+args['centroids'], :] , axis=0))
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
    info = ExperimentInfo() 
    for i in range(len(data)):
        Xs = np.delete(data[i], np.s_[1::2], 1)
        Ys = np.delete(data[i], np.s_[0::2], 1)
        minX = np.min(Xs)
        maxX = np.max(Xs)
        minY = np.min(Ys)
        maxY = np.max(Ys)
        if info.minX < minX:
            info.minX = minX
        if info.maxX > maxX:
            info.maxX = maxX
        if info.minY < minY:
            info.minY = minY
        if info.maxY > maxY:
            info.maxY = maxY

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
    print('Lines skipped ' + str(data.shape[0] - filtered_data.shape[0]) + ' out of ' + str(data.shape[0]))    
    return filtered_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the positions')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--filename', '-f', type=str,
                        help='Position file name',
                        required=True)
    args = parser.parse_args()

    data, info = filter(load(args.path, args.filename), skip_zero_movement, 
        args={
            'invertY' : True,
            'resY' : 1500,
            'scale' : 1.11111111111 / 1500,
            'initial_keep' : 104400,
            'centroids' : 3, 
            'eps': 0.00016
        })


    # print(info.center())