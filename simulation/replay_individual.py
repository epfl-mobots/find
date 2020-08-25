from simulation.simu.simulation.individual import Individual

class ReplayIndividual(Individual):
    def __init__(self, positions, velocities):
        super().__init__(is_robot=False) # explicitly setting that this is not a robotic/virtual individual
        self._position_history = positions
        self._velocity_history = velocities
        self._position = self._position_history[0, :]
        self._velocity = self._velocity_history[0, :]

    def interact(self, simu):
        pass

    def move(self, simu):
        self._position = self._position_history[simu.get_current_iteration(), :]
        self._velocity = self._velocity_history[simu.get_current_iteration(), :]

    def _history_update(self, simu):
        pass