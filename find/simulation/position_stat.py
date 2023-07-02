from find.simulation.simu.stat.stat_base import StatBase

import numpy as np
from pathlib import Path


class PositionStat(StatBase):
    def __init__(self, dims, filename, dirname='', dump_period=-1):
        super().__init__(filename, dirname, dump_period)
        self._positions = np.empty((0, dims))
        self._dims = dims

    def get_filename(self):
        return self._filename

    def get(self):
        return self._positions

    def save(self):
        np.savetxt(Path(self._dirname).joinpath(
            self._filename), self._positions)

    def __call__(self, simu):
        early_dump = self._dump_period > 0 and simu.get_current_iteration() % self._dump_period == 0

        if simu.get_num_iterations() == simu.get_current_iteration() + 1 or early_dump:
            appended_pos = np.empty((1, 0))
            for ind in simu.get_individuals():
                cpos = ind.get_position_history(
                )[simu.get_current_iteration(), :].reshape(-1, 2)
                appended_pos = np.hstack((appended_pos, cpos))
            self._positions = np.vstack((self._positions, appended_pos))

            if early_dump:
                self.save()
