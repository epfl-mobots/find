from simulation.simu.stat.stat_base import StatBase

import numpy as np
from pathlib import Path

class VelocityStat(StatBase):
    def __init__(self, dims, filename, dirname=''):
        super().__init__(filename, dirname)
        self._velocities = np.empty((0, dims))
        self._dims = dims

    def get_filename(self):
        return self._filename

    def get(self):
        return self._velocities

    def save(self):
        np.savetxt(Path(self._dirname).joinpath(self._filename), self._velocities)

    def __call__(self, simu):
        if simu.get_num_iterations() == simu.get_current_iteration() + 1:
            appended_vel = np.empty((simu.get_individuals()[0].get_velocity_history().shape[0], 0))
            for ind in simu.get_individuals():
                appended_vel = np.hstack((appended_vel, ind.get_velocity_history()))
            self._velocities = appended_vel