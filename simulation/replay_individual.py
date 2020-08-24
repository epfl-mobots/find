from simulation.simu.simulation.individual import Individual

class ReplayIndividual(Individual):
    def __init__(self, positions, velocities):
        super().__init__(is_robot=False) # explicitly setting that this is not a robotic/virtual individual
        self._positions = positions
        self._velocities = velocities

    def interact(self, simu):
        pass

    def move(self, simu):
        self._position = self._positions[simu.get_current_iteration(), :]
        self._velocity = self._velocities[simu.get_current_iteration(), :]