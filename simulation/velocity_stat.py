from simulation.simu.simulation.stat import StatBase

import numpy as np

class PositionStat(StatBase):
    def __init__(self, filename):
        super().__init__(filename)
        self._positions = np.empty()

    def __call__(self):
        pass
        
