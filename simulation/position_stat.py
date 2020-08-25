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
        if simu.get_num_iterations() == simu.get_current_iteration() + 1:
            appended_pos = np.empty((simu.get_individuals()[0].get_position_history().shape[0], 0))
            for ind in simu.get_individuals():
                appended_pos = np.hstack((appended_pos, ind.get_position_history()))
            self._positions = appended_pos