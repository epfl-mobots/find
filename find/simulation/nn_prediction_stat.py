from find.simulation.simu.stat.stat_base import StatBase

import numpy as np
from pathlib import Path


class NNPredictionStat(StatBase):
    def __init__(self, nn_functor, dims, filename, dirname=''):
        super().__init__(filename, dirname)
        self._positions = np.empty((0, dims))
        self._dims = dims
        self._nn_functor = nn_functor

    def get_filename(self):
        return self._filename

    def get(self):
        return self._positions

    def save(self):
        np.savetxt(Path(self._dirname).joinpath(
            self._filename), self._positions)

    def __call__(self, simu):
        if simu.get_current_iteration() > 1:
            individuals = simu.get_individuals()
            row = np.empty((1, self._dims))
            for ind in individuals:
                if ind.is_robot():
                    pos = ind.get_position()
                else:
                    pos = self._nn_functor(ind.get_id(), simu)
                row = np.hstack((row, pos.reshape(1, -1)))
            self._positions = np.vstack((self._positions, row))
