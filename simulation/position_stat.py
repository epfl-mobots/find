from simulation.simu.stat.stat_base import StatBase

import numpy as np
from pathlib import Path

class PositionStat(StatBase):
    def __init__(self, dims, filename, dirname=''):
        super().__init__(filename, dirname)
        self._positions = np.empty((0, dims))
        self._dims = dims

    def get_filename(self):
        return self._filename

    def get(self):
        return self._positions

    def save(self):
        np.savetxt(Path(self._dirname).joinpath(self._filename), self._positions)

    def __call__(self, simu):
        iter_pos = np.empty((1, 0))
        for ind in simu.get_individuals():
            iter_pos = np.hstack((iter_pos, ind.get_position().reshape(1, -1)))
        self._positions = np.vstack((self._positions, iter_pos))