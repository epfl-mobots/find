import numpy as np


class Velocities:
    """Simplistic instantaneous velocity computation."""

    def __init__(self, positions, timestep):
        """
        :param positions: list(np.array) of (x_1, y_1, ..., x_n, y_n) position matrices of fish individuals
        :param timestep: float time interval between each measurement (sampling rate)
        """
        self._velocities = []
        for p in positions:
            rolled_p = np.roll(p, shift=1, axis=0)
            velocities = (p - rolled_p) / timestep
            sigma = np.std(velocities[1:, :], axis=0)
            mu = np.mean(velocities[1:, :], axis=0)
            x_rand = np.random.normal(mu[0], sigma[0], p.shape[1] // 2)[0]
            y_rand = np.random.normal(mu[1], sigma[1], p.shape[1] // 2)[0]
            for i in range(p.shape[1] // 2):
                velocities[0, i * 2] = velocities[1, i * 2] + x_rand
                velocities[0, i * 2 + 1] = velocities[1, i * 2] + y_rand
            self._velocities.append(velocities)

    def get(self):
        """
        :return: list(np.array) of resultant velocities for each of the matrices provided to the class
        """
        return self._velocities
