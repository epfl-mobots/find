import numpy as np


class Derivative:
    def __init__(self, matrix, timestep):
        """
        :param positions: list(np.array) of (x_1, y_1, ..., x_n, y_n) position matrices of fish individuals
        :param timestep: float time interval between each measurement (sampling rate)
        """
        self._deriv = []
        for m in matrix:
            rolled_m = np.roll(m, shift=1, axis=0)
            deriv = (m - rolled_m) / timestep
            sigma = np.std(deriv[1:, :], axis=0)
            mu = np.mean(deriv[1:, :], axis=0)
            x_rand = np.random.normal(mu[0], sigma[0], m.shape[1] // 2)[0]
            y_rand = np.random.normal(mu[1], sigma[1], m.shape[1] // 2)[0]
            for i in range(m.shape[1] // 2):
                deriv[0, i * 2] = deriv[1, i * 2] + x_rand
                deriv[0, i * 2 + 1] = deriv[1, i * 2] + y_rand
            self._deriv.append(deriv)

    def get(self):
        """
        :return: list(np.array) of resultant derivatives for each of the matrices provided to the class
        """
        return self._deriv


class Velocities(Derivative):
    """Simplistic instantaneous velocity computation."""

    def __init__(self, positions, timestep):
        """
        :param positions: list(np.array) of (x_1, y_1, ..., x_n, y_n) position matrices of fish individuals
        :param timestep: float time interval between each measurement (sampling rate)
        """
        super().__init__(positions, timestep)

    def get(self):
        """
        :return: list(np.array) of resultant velocities for each of the matrices provided to the class
        """
        return super().get()


class Accelerations(Derivative):
    """Simplistic instantaneous accelaration computation."""

    def __init__(self, velocities, timestep):
        """
        :param velocities: list(np.array) of (x_1, y_1, ..., x_n, y_n) position matrices of fish individuals
        :param timestep: float time interval between each measurement (sampling rate)
        """
        super().__init__(velocities, timestep)

    def get(self):
        """
        :return: list(np.array) of resultant acceleration for each of the matrices provided to the class
        """
        return super().get()
