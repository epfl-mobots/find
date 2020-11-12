from find.simulation.simu.simulation.simulation import Simulation


class FishSimulation(Simulation):
    def __init__(self, timestep, num_iterations, args={'stats_enabled': False}):
        # TODO: super() is python 3 specific
        super().__init__(timestep, num_iterations, args)
