import numpy as np

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


class Center:
    def __init__(self, data, info, args={}):
        for i, matrix in enumerate(data):
            c = info.center(i)
            for n in range(matrix.shape[1] // 2):
                matrix[:, n * 2] =  matrix[:, n * 2] - c[0] 
                matrix[:, n * 2 + 1] = matrix[:, n * 2 + 1] - c[1]
        self._data = data
        self._info = ExperimentInfo(data) 


    def get(self):
        return self._data, self._info


class Normalize: 
    def __init__(self, data, info, args={'is_circle' : True}):
        for i, matrix in enumerate(data):
            if args.is_circle:
                for n in range(matrix.shape[1] // 2):
                    rads = matrix
                    rads[:, n * 2] -= info.center()[0]
                    rads[:, n * 2 + 1] -= info.center()[1]
                    phis = np.arctan2(rads[:, n * 2 + 1], rads[:, n * 2])
                    rads[:, n * 2] = rads[:, n * 2] ** 2
                    rads[:, n * 2 + 1] = rads[:, n * 2 + 1] ** 2
                    rads = np.sqrt(rads[:, n * 2] + rads[:, n * 2 + 1])
                    rads /= np.max(rads)
                    matrix[:, n * 2] = rads * np.cos(phis)
                    matrix[:, n * 2 + 1] = rads * np.sin(phis)
            else:
                maxXY = info.maxXY(i)
                for n in range(matrix.shape[1] // 2):
                    matrix[:, n * 2] /= maxXY[0] 
                    matrix[:, n * 2 + 1] /= maxXY[1]
        self._data = data
        self._info = ExperimentInfo(data) 


    def get(self):
        return self._data, self._info