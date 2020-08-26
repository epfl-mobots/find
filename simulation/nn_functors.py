import numpy as np
from random import shuffle

class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius < self._radius and radius > 0

    def center(self):
        return self._center

setup = CircularCorridor()

velocity_threshold = 1.4

def logbound(val, max_logvar=0, min_logvar=-10):
    logsigma = max_logvar - np.log(np.exp(max_logvar - val) + 1)
    logsigma = min_logvar + np.log(np.exp(logsigma - min_logvar) + 1)
    return logsigma
    

def _sample_valid_velocity(position, prediction, timestep):
    failed = 0
    (x_hat, y_hat) = (None, None)

    while True:
        sample_velx = np.random.normal(prediction[0, 0], prediction[0, 2], 1)[0]
        sample_vely = np.random.normal(prediction[0, 1], prediction[0, 3], 1)[0]

        x_hat = position[0] + sample_velx * timestep
        y_hat = position[1] + sample_vely * timestep
        r = np.sqrt((x_hat - setup.center()[0]) ** 2 + (y_hat - setup.center()[1]) ** 2)

        rv = np.sqrt(sample_velx ** 2 +
                    sample_vely ** 2 -
                    2 * np.abs(sample_velx) * np.abs(sample_vely) * np.cos(np.arctan2(sample_vely, sample_velx)))

        if setup.is_valid(r):
            return np.array([x_hat, y_hat])
        else:
            failed += 1
            if failed > 999:
                prediction[:, 2] += 0.01
                prediction[:, 3] += 0.01

    return np.array([x_hat, y_hat])


class Multi_pfw_predict:
    def __init__(self, model):
        self._model = model
        
    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]
        
        X = [
                focal.get_position()[0], 
                focal.get_position()[1], 
                focal.get_velocity()[0], 
                focal.get_velocity()[1]]

        ind_ids = list(range(len(individuals)))
        shuffle(ind_ids)
        for idx in ind_ids:
            ind = individuals[idx]
            if ind.get_id() == focal_id:
                continue
            X = X + [
                    ind.get_position()[0], 
                    ind.get_position()[1], 
                    ind.get_velocity()[0], 
                    ind.get_velocity()[1]]

        X = np.array(X)

        prediction = np.array(self._model.predict(X.reshape(1, X.shape[0])))
        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

        return _sample_valid_velocity(focal.get_position(), prediction, simu.get_timestep())


class Multi_plstm_predict:
    def __init__(self, model, num_timesteps):
        self._model = model
        self._num_timesteps = num_timesteps
        
    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]
        
        X = np.empty((self._num_timesteps, 0))
        
        p = focal.get_position_history()
        v = focal.get_velocity_history()
        X = np.hstack((X, p[-self._num_timesteps:, :]))
        X = np.hstack((X, v[-self._num_timesteps:, :]))

        ind_ids = list(range(len(individuals)))
        shuffle(ind_ids)
        for idx in ind_ids:
            ind = individuals[idx]
            if ind.get_id() == focal_id:
                continue
            p = ind.get_position_history()
            v = ind.get_velocity_history()
            X = np.hstack((X, p[-self._num_timesteps:, :]))
            X = np.hstack((X, v[-self._num_timesteps:, :]))

        prediction = np.array(self._model.predict(X.reshape(1, self._num_timesteps, X.shape[1])))
        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

        return _sample_valid_velocity(focal.get_position(), prediction, simu.get_timestep())
