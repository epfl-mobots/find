import numpy as np

class Individual:
    _ind_id = 0

    def __init__(self, is_robot=False):
        self._is_robot = is_robot
        self._id = Individual._ind_id
        Individual._ind_id += 1
        self._position = None
        self._velocity = None
        self._position_history = None
        self._velocity_history = None

    def get_id(self):
        return self._id

    def get_position(self):
        return self._position

    def get_velocity(self):
        return self._velocity

    def set_position(self, pos):
        self._position = pos

    def set_velocity(self, vel):
        self._velocity = vel

    def get_position_history(self):
        return self._position_history

    def get_velocity_history(self):
        return self._velocity_history

    def set_position_history(self, posh):
        self._position_history = posh
        self._position = self._position_history[-1, :]

    def set_velocity_history(self, velh):
        self._velocity_history = velh
        self._velocity = self._velocity_history[-1, :]

    def is_robot(self):
        return self._is_robot

    def run(self, simu):
        self.interact(simu)
        self.move(simu)
        self._history_update(simu)


    def interact(self, simu):
        assert False, 'You need to implement this function in a subclass'

    def move(self, simu):
        assert False, 'You need to implement this function in a subclass'

    def _history_update(self, simu):
        if self._position_history is None:
            self._position_history = np.empty((0, len(self._position)))
            self._velocity_history = np.empty((0, len(self._velocity)))
        else:
            self._position_history = np.vstack((self._position_history, self._position.reshape(1, -1)))
            self._velocity_history = np.vstack((self._velocity_history, self._velocity.reshape(1, -1)))
