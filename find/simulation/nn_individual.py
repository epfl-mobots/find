import numpy as np

from find.simulation.simu.simulation.individual import Individual


class NNIndividual(Individual):
    def __init__(self, query_func, initial_pos=(0, 0), initial_vel=(0, 0)):
        # explicitly setting that this is not a robotic/virtual individual
        super().__init__(is_robot=True)
        self._query_func = query_func

        if initial_pos.ndim > 1:
            self._position_history = np.array(initial_pos)
            self._velocity_history = np.array(initial_vel)
            self._position = self._position_history[-1, :]
            self._velocity = self._velocity_history[-1, :]
        else:
            self._position = np.array(initial_pos)
            self._velocity = np.array(initial_vel)
            self._position_history = np.empty((0, len(self._position)))
            self._velocity_history = np.empty((0, len(self._velocity)))
            self._position_history = np.vstack(
                (self._position_history, self._position.reshape(1, -1)))
            self._velocity_history = np.vstack(
                (self._velocity_history, self._velocity.reshape(1, -1)))
            self._next_position = None
            self._next_velocity = None

    def interact(self, simu):
        # this should always be expressed in the next position TODO: maybe generalize ?
        self._next_position = self._query_func(self._id, simu)
        if type(self._next_position) is not list:
            self._next_velocity = (self._next_position -
                                   self._position) / simu.get_timestep()
        else:
            self._next_velocity = None

    def move(self, simu):
        if type(self._next_position) is list:
            np = self._next_position
            for i in range(len(self._next_position)-1):
                self._next_position = np[i]
                self._velocity = (self._next_position -
                                  self._position) / simu.get_timestep()
                self._update_history(simu)
            self._next_position = np[-1]
            self._velocity = (self._next_position -
                              self._position) / simu.get_timestep()
        else:
            self._position = self._next_position
            self._velocity = self._next_velocity
