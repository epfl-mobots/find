from find.simulation.simu.simulation.simulation import Simulation
from find.simulation.simu.simulation.individual import Individual


class FishSimulation(Simulation):
    def __init__(self, timestep, num_iterations, args={'stats_enabled': False}):
        Individual.reset_ind_id()
        # TODO: super() is python 3 specific
        super().__init__(timestep, num_iterations, args)
