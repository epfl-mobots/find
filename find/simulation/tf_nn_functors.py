import numpy as np
from random import shuffle
from find.utils.losses import logbound


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def is_valid(self, radius):
        return radius < self._radius and radius > 0

    def center(self):
        return self._center


setup = CircularCorridor()


def _sample_valid_position(position, velocity, prediction, timestep, args):
    failed = 0
    (x_hat, y_hat) = (None, None)

    while True:
        sample_velx = np.random.normal(
            prediction[0, 0], np.sqrt(prediction[0, 2]) * args.var_coef, 1)[0]
        sample_vely = np.random.normal(
            prediction[0, 1], np.sqrt(prediction[0, 3]) * args.var_coef, 1)[0]

        vx_hat = velocity[0] + sample_velx
        vy_hat = velocity[1] + sample_vely
        x_hat = position[0] + vx_hat * timestep
        y_hat = position[1] + vy_hat * timestep
        r = np.sqrt((x_hat - setup.center()[0])
                    ** 2 + (y_hat - setup.center()[1]) ** 2)

        if setup.is_valid(r):
            return np.array([x_hat, y_hat])
        else:
            failed += 1
            if failed > 999:
                prediction[:, 2] += 0.01
                prediction[:, 3] += 0.01

    return np.array([x_hat, y_hat])


def closest_individual(focal_id, individuals):
    focal_idx = None
    for i, ind in enumerate(individuals):
        if ind.get_id() == focal_id:
            focal_idx = i
            break

    distance = []
    fpos = individuals[focal_id].get_position()
    for ind in individuals:
        pos = ind.get_position()
        distance.append(np.sqrt((pos[0] - fpos[0])
                                ** 2 + (pos[1] - fpos[1]) ** 2))
    ind_idcs = [x for _, x in sorted(
        zip(distance, list(range(len(individuals)))))]
    ind_idcs.remove(focal_idx)
    return ind_idcs


def shuffled_individuals(focal_id, individuals):
    ind_ids = list(range(len(individuals)))
    focal_idx = None
    for i, ind in enumerate(individuals):
        if ind.get_id() == focal_id:
            focal_idx = i
            break
    shuffle(ind_ids)
    return ind_ids


most_influential_individual = {
    'closest': closest_individual,
    'shuffled': shuffled_individuals,
}


def get_most_influential_individual():
    return list(most_influential_individual.keys())


class Multi_pfw_predict:
    def __init__(self, model, args, num_neighs=1):
        self._model = model
        self._num_neighs = num_neighs
        self._selection = most_influential_individual[args.most_influential_individual]
        self._args = args

    def _compute_dist_wall(self, p):
        rad = 1 - np.array(np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)).T
        zidcs = np.where(rad < 0)
        if len(zidcs[0]) > 0:
            rad[zidcs] = 0
        return rad

    def _compute_inter_dist(self, p1, p2):
        return np.sqrt((p1[:, 0] - p2[:, 2]) ** 2 + (p1[:, 1] - p2[:, 3]) ** 2)

    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]

        X = [
            focal.get_position()[0],
            focal.get_position()[1],
            focal.get_velocity()[0],
            focal.get_velocity()[1]]
        if self._args.distance_inputs:
            X += list(self._compute_dist_wall(focal.get_position()))

        ind_idcs = self._selection(focal_id, individuals)
        for idx in ind_idcs[:self._num_neighs]:
            ind = individuals[idx]
            X = X + [
                ind.get_position()[0],
                ind.get_position()[1],
                ind.get_velocity()[0],
                ind.get_velocity()[1]]
            if self._args.distance_inputs:
                X += list(self._compute_inter_dist(
                    focal.get_position(),
                    ind.get_position()))
        X = np.array(X)

        prediction = np.array(self._model.predict(X.reshape(1, X.shape[0])))
        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

        return _sample_valid_position(focal.get_position(), focal.get_velocity(), prediction, simu.get_timestep(), self._args)


class Multi_plstm_predict:
    def __init__(self, model, num_timesteps, args, num_neighs=1):
        self._model = model
        self._num_timesteps = num_timesteps
        self._num_neighs = num_neighs
        self._selection = most_influential_individual[args.most_influential_individual]
        self._args = args

    def _compute_dist_wall(self, p):
        rad = 1 - np.array(np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)).T
        zidcs = np.where(rad < 0)
        if len(zidcs[0]) > 0:
            rad[zidcs] = 0
        return rad

    def _compute_inter_dist(self, p1, p2):
        return np.sqrt((p1[:, 0] - p2[:, 2]) ** 2 + (p1[:, 1] - p2[:, 3]) ** 2)

    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]

        X = np.empty((self._num_timesteps, 0))

        p1 = focal.get_position_history()
        v1 = focal.get_velocity_history()
        X = np.hstack((X, p1[-self._num_timesteps:, :]))
        X = np.hstack((X, v1[-self._num_timesteps:, :]))
        if self._args.distance_inputs:
            rad = self._compute_dist_wall(p1[-self._num_timesteps:, :])
            X = np.hstack((X, rad))

        ind_idcs = self._selection(focal_id, individuals)
        for idx in ind_idcs[: self._num_neighs]:
            ind = individuals[idx]
            p2 = ind.get_position_history()
            v2 = ind.get_velocity_history()
            X = np.hstack((X, p2[-self._num_timesteps:, :]))
            X = np.hstack((X, v2[-self._num_timesteps:, :]))
            if self._args.distance_inputs:
                dist = self._compute_inter_dist(
                    p1[-self._num_timesteps:, :],
                    p2[-self._num_timesteps:, :])
                X = np.hstack((X, dist))

        prediction = np.array(self._model.predict(
            X.reshape(1, self._num_timesteps, X.shape[1])))
        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

        return _sample_valid_position(focal.get_position(), focal.get_velocity(), prediction, simu.get_timestep(), self._args)


class Multi_plstm_predict_traj:
    def __init__(self, model, num_timesteps, args, num_neighs=1):
        self._model = model
        self._num_timesteps = num_timesteps
        self._num_neighs = num_neighs
        self._selection = most_influential_individual[args.most_influential_individual]
        self._args = args

    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]

        X = np.empty((self._num_timesteps, 0))

        p1 = focal.get_position_history()
        v1 = focal.get_velocity_history()
        X = np.hstack((X, p1[-self._num_timesteps:, :]))
        X = np.hstack((X, v1[-self._num_timesteps:, :]))
        if self._args.distance_inputs:
            rad = self._compute_dist_wall(p1[-self._num_timesteps:, :])
            X = np.hstack((X, rad))

        ind_idcs = self._selection(focal_id, individuals)
        for idx in ind_idcs[:self._num_neighs]:
            ind = individuals[idx]
            p2 = ind.get_position_history()
            v2 = ind.get_velocity_history()
            X = np.hstack((X, p2[-self._num_timesteps:, :]))
            X = np.hstack((X, v2[-self._num_timesteps:, :]))
            if self._args.distance_inputs:
                dist = self._compute_inter_dist(
                    p1[-self._num_timesteps:, :],
                    p2[-self._num_timesteps:, :])
                X = np.hstack((X, dist))

        prediction = np.array(self._model.predict(
            X.reshape(1, self._num_timesteps, X.shape[1])))

        valid_predictions = []
        for i in range(prediction.shape[2]):
            pri = prediction[0, 0, i, :].reshape(1, prediction.shape[3])
            pri[0, 2:] = list(map(logbound, pri[0, 2:]))
            pri[0, 2:] = list(map(np.exp, pri[0, 2:]))
            valid_predictions.append(_sample_valid_position(focal.get_position(),
                                                            pri, simu.get_timestep()), self._args)

        return valid_predictions
