import numpy as np


class ExperimentInfo:
    """Container class for a single experiment's information. This is useful for aligning all the experiments
       in terms of the setup position, etc.
    """

    def __init__(self, data):
        """
        :param data: np.array of the raw iDTracker data (this includes confidence values for the fish positions)
        """
        self._maxXs = [np.max(np.delete(matrix, np.s_[1::2], 1)) for matrix in data]
        self._minXs = [np.min(np.delete(matrix, np.s_[1::2], 1)) for matrix in data]
        self._global_minX = np.min(self._minXs)
        self._global_maxX = np.max(self._maxXs)

        self._maxYs = [np.max(np.delete(matrix, np.s_[0::2], 1)) for matrix in data]
        self._minYs = [np.min(np.delete(matrix, np.s_[0::2], 1)) for matrix in data]
        self._global_minY = np.min(self._minYs)
        self._global_maxY = np.max(self._maxYs)

    def center(self, idx=-1):
        """
        :param idx: int, optional index of the individual's trajectories that will be used as reference for determining
                                  the setup's center. If -1 the center will be computed by looking at all individuals.
        :return: tuple(float, float) of the center coordinates
        """
        if idx < 0:
            return ((self._global_maxX + self._global_minX) / 2, (self._global_maxY + self._global_minY) / 2)
        else:
            return ((self._maxXs[idx] + self._minXs[idx]) / 2, (self._maxYs[idx] + self._minYs[idx]) / 2)

    def minXY(self, idx):
        """
        :param idx: int index of the individual's trajectories
        :return: tuple(float, float) minimum values for X and Y
        """
        return (self._minXs[idx], self._minYs[idx])

    def maxXY(self, idx):
        """
        :param idx: int index of the individual's trajectories
        :return: tuple(float, float) maximum values for X and Y
        """
        return (self._maxXs[idx], self._maxYs[idx])

        """
        :return: tuple(float, float) global minimum values for X and Y
        """
    def globalMinXY(self):
        return (self._global_minX, self._global_minY)
    
        """
        :return: tuple(float, float) global maximum values for X and Y
        """
    def globalMaxXY(self):
        return (self._global_maxX, self._global_maxY)

    def printInfo(self):
        print('Center: ' + str(self.center()))
        print('min(X, Y): ' + str(self._global_minX) + ', ' + str(self._global_minY))
        print('max(X, Y): ' + str(self._global_maxX) + ', ' + str(self._global_maxY))


class Center:
    """Class responsible for centering the experimental data to (0, 0). This is especially useful in cases were
       the experimental setup might slightly (or even significantly) move from one experiment to the next.
    """

    def __init__(self, data, info, args={}):
        """
        :param data:  list(np.array) of matrices with (position) information
        :param info: ExperimentInfo instance for the given
        :param args: dict, optional extra arguments for the function (not applicable)
        """
        for i, matrix in enumerate(data):
            c = info.center()
            for n in range(matrix.shape[1] // 2):
                matrix[:, n * 2] = matrix[:, n * 2] - c[0]
                matrix[:, n * 2 + 1] = matrix[:, n * 2 + 1] - c[1]
        self._data = data
        self._info = ExperimentInfo(data)


    def get(self):
        """
        :return: tuple(list(np.array), ExperimentInfo) the centered positions and updated experiment information
        """
        return self._data, self._info


class Normalize:
    """Class responsible for normalizing the experimental data in the range [-1, 1]. Different experiments might have
       slightly different boundaries due to changes in the setup's position. The normalization process allows for
       bringing all experiments in the same range and comparing the behavioural side without numerical issues.
    """

    def __init__(self, data, info, args={'is_circle': True}):
        """
        :param data:  list(np.array) of matrices with (position) information
        :param info: ExperimentInfo instance for the given
        :param args: dict, optional extra arguments for the function (not applicable)
        """

        for i, matrix in enumerate(data):
            radius = np.max([info.globalMaxXY()[0]-info.globalMinXY()[0], info.globalMaxXY()[1]-info.globalMinXY()[1]]) / 2
            if args['is_circle']:
                for n in range(matrix.shape[1] // 2):
                    rads = matrix
                    rads[:, n * 2] -= info.center()[0]
                    rads[:, n * 2 + 1] -= info.center()[1]
                    phis = np.arctan2(rads[:, n * 2 + 1], rads[:, n * 2])
                    rads[:, n * 2] = rads[:, n * 2] ** 2
                    rads[:, n * 2 + 1] = rads[:, n * 2 + 1] ** 2
                    rads = np.sqrt(rads[:, n * 2] + rads[:, n * 2 + 1])
                    rads /= radius
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
        """
        :return: tuple(list(np.array), ExperimentInfo) the centered positions and updated experiment information
        """
        return self._data, self._info


def angle_to_pipi(dif):
    """
    :param dif: float angle difference between the heading of two individuals
    :return: float smallest difference within the range of -pi and pi
    """
    while True:
        if dif < -np.pi:
            dif += 2. * np.pi
        if dif > np.pi:
            dif -= 2. * np.pi
        if (np.abs(dif) <= np.pi):
            break
    return dif
