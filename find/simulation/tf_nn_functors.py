import numpy as np
from random import shuffle
from find.models.tf_losses import logbound


class CircularCorridor:
    def __init__(self, radius=1.0, center=(0, 0)):
        self._center = center
        self._radius = radius

    def radius(self, position):
        return np.sqrt((position[0] - self._center[0]) ** 2 + (position[1] - self._center[1]) ** 2)

    def is_valid(self, radius):
        return radius < self._radius and radius > 0

    def center(self):
        return self._center


setup = CircularCorridor()


def _sample_valid_position(position, velocity, prediction, timestep, args):
    failed = 0
    (x_hat, y_hat) = (None, None)

    while True:
        g_x = np.random.normal(
            prediction[0, 0], prediction[0, 2] * args.var_coef, 1)[0]
        g_y = np.random.normal(
            prediction[0, 1], prediction[0, 3] * args.var_coef, 1)[0]

        vx_hat = velocity[0] + g_x
        vy_hat = velocity[1] + g_y
        x_hat = position[0] + vx_hat * timestep
        y_hat = position[1] + vy_hat * timestep
        r = np.sqrt((x_hat - setup.center()[0])
                    ** 2 + (y_hat - setup.center()[1]) ** 2)
        dist = np.sqrt((x_hat - position[0])
                       ** 2 + (y_hat - position[1]) ** 2)

        # if setup.is_valid(r):  # and dist <= args.body_len / 2:
        if setup.is_valid(r) and dist <= 0.2:
            return np.array([x_hat, y_hat])
        else:
            failed += 1
            if failed > 999:
                prediction[:, 2] += 0.01
                prediction[:, 3] += 0.01


class ClosestIndividual:
    def __init__(self, args):
        self._args = args

    def sort(self, focal_id, individuals, simu):
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

    def select(self, focal_id, predictions, simu):
        return predictions[0]


class ShuffledIndividuals:
    def __init__(self, args):
        self._args = args

    def sort(self, focal_id, individuals, simu):
        ind_ids = list(range(len(individuals)))
        focal_idx = None
        for i, ind in enumerate(individuals):
            if ind.get_id() == focal_id:
                focal_idx = i
                break
        ind_ids.remove(focal_idx)
        shuffle(ind_ids)
        return ind_ids

    def select(self, focal_id, predictions, simu):
        pass


class HighestAcceleration:
    def __init__(self, args):
        self._args = args

    def sort(self, focal_id, individuals, simu):
        ind_ids = list(range(len(individuals)))
        focal_idx = None
        for i, ind in enumerate(individuals):
            if ind.get_id() == focal_id:
                focal_idx = i
                break
        ind_ids.remove(focal_idx)
        return ind_ids

    def select(self, focal_id, predictions, simu):
        inds = simu.get_individuals()
        minf_vec = []
        # vels = []
        # accs = []
        preds = []
        # for i in range(len(inds)):
        #     if inds[i].get_id() == focal_id:
        #         continue
        #     v = inds[i].get_velocity()
        #     vels.append(np.sqrt(v[0] ** 2 + v[1] ** 2))

        #     a = inds[i].get_acceleration()
        #     if a is not None:
        #         accs.append(np.sqrt(a[0] ** 2 + a[1] ** 2))

        for p in predictions:
            preds.append(np.sqrt(p[0, 2] ** 2 + p[0, 3] ** 2))

        # if len(accs) == len(vels):
        #     minf_vec = accs
        # else:
        #     minf_vec = vels
        # minf_vec = accs
        # minf_vec = vels
        minf_vec = preds

        sorted_idcs = sorted(
            range(len(minf_vec)),
            key=lambda index: minf_vec[index],
            reverse=True
        )
        new_pred = predictions[sorted_idcs[0]]

        for i in range(1, self._args.num_neighs_consider):
            ind = sorted_idcs[i]
            new_pred[0, 0] += predictions[ind][0, 0]
            new_pred[0, 1] += predictions[ind][0, 1]
            new_pred[0, 2] += ((predictions[ind][0, 2]
                                * self._args.var_coef) ** 2)
            new_pred[0, 3] += ((predictions[ind][0, 3]
                                * self._args.var_coef) ** 2)
        new_pred[0, 0] /= self._args.num_neighs_consider
        new_pred[0, 1] /= self._args.num_neighs_consider
        new_pred[0, 2] = np.sqrt(
            new_pred[0, 2]) / (self._args.num_neighs_consider)
        new_pred[0, 3] = np.sqrt(
            new_pred[0, 2]) / (self._args.num_neighs_consider)

        return new_pred


most_influential_individual = {
    'closest': ClosestIndividual,
    'shuffled': ShuffledIndividuals,
    'highest_acc': HighestAcceleration
}


def get_most_influential_individual():
    return list(most_influential_individual.keys())


class Multi_pfw_predict:
    def __init__(self, model, args, num_neighs=1):
        self._model = model
        self._num_neighs = num_neighs
        self._selection_method = most_influential_individual[args.most_influential_individual](
            args)
        self._args = args

    def _compute_dist_wall(self, p):
        rad = 1 - np.array(np.sqrt(p[0] ** 2 + p[1] ** 2)).T
        zidcs = np.where(rad < 0)
        if len(zidcs[0]) > 0:
            rad[zidcs] = 0
        return rad

    def _compute_inter_dist(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]

        X = [
            focal.get_position()[0],
            focal.get_position()[1],
            focal.get_velocity()[0],
            focal.get_velocity()[1]]
        if self._args.distance_inputs:
            X.append(self._compute_dist_wall(focal.get_position()))

        ind_idcs = self._selection_method.sort(
            focal_id, individuals, simu, simu)
        for idx in ind_idcs:
            ind = individuals[idx]
            X = X + [
                ind.get_position()[0],
                ind.get_position()[1],
                ind.get_velocity()[0],
                ind.get_velocity()[1]]
            if self._args.distance_inputs:
                X.append(self._compute_dist_wall(ind.get_position()))
                X.append(self._compute_inter_dist(
                    focal.get_position(),
                    ind.get_position()))
        X = np.array(X)

        prediction = np.array(self._model.predict(X.reshape(1, X.shape[0])))
        prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
        prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))

        prediction = self._selection_method.select(focal_id, predictions, simu)
        return _sample_valid_position(focal.get_position(), focal.get_velocity(), prediction, simu.get_timestep(), self._args)


class Multi_plstm_predict:
    def __init__(self, model, num_timesteps, args, num_neighs=1):
        self._model = model
        self._num_timesteps = num_timesteps
        self._num_neighs = num_neighs
        self._selection_method = most_influential_individual[args.most_influential_individual](
            args)
        self._args = args
        self._means = [None]
        self._stds = [None]

    def get_means(self):
        return self._means

    def get_stds(self):
        return self._stds

    def _compute_dist_wall(self, p):
        rad = 1 - np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2).T
        zidcs = np.where(rad < 0)
        if len(zidcs[0]) > 0:
            rad[zidcs] = 0
        return rad

    def _compute_inter_dist(self, p1, p2):
        return np.sqrt((p1[:, 0] - p2[:, 0]) ** 2 + (p1[:, 1] - p2[:, 1]) ** 2)

    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        if self._means[0] is None:
            self._means = [None] * len(simu.get_individuals())
            self._stds = [None] * len(simu.get_individuals())

        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]

        X = np.empty((self._num_timesteps, 0))

        p1 = focal.get_position_history()
        v1 = focal.get_velocity_history()
        X = np.hstack((X, p1[-self._num_timesteps:, :]))
        X = np.hstack((X, v1[-self._num_timesteps:, :]))
        if self._args.distance_inputs:
            rad = self._compute_dist_wall(p1[-self._num_timesteps:, :])
            X = np.hstack((X, rad.reshape(-1, 1)))

        predictions = []

        ind_idcs = self._selection_method.sort(focal_id, individuals, simu)
        for idx in ind_idcs:
            ind = individuals[idx]
            p2 = ind.get_position_history()
            v2 = ind.get_velocity_history()
            Xhat = np.hstack((X, p2[-self._num_timesteps:, :]))
            Xhat = np.hstack((Xhat, v2[-self._num_timesteps:, :]))
            if self._args.distance_inputs:
                rad = self._compute_dist_wall(p2[-self._num_timesteps:, :])
                Xhat = np.hstack((Xhat, rad.reshape(-1, 1)))

                dist = self._compute_inter_dist(
                    p1[-self._num_timesteps:, :],
                    p2[-self._num_timesteps:, :])
                Xhat = np.hstack((Xhat, dist.reshape(-1, 1)))

            prediction = np.array(self._model.predict(
                Xhat.reshape(1, self._num_timesteps, Xhat.shape[1])))
            prediction[0, 2:] = list(map(logbound, prediction[0, 2:]))
            prediction[0, 2:] = list(map(np.exp, prediction[0, 2:]))
            predictions.append(prediction)

        prediction = self._selection_method.select(focal_id, predictions, simu)
        self._means[focal_id] = np.array([
            focal.get_position()[0] + (focal.get_velocity()
                                       [0] + prediction[0, 0]) * simu.get_timestep(),
            focal.get_position()[1] + (focal.get_velocity()
                                       [1] + prediction[0, 1]) * simu.get_timestep()
        ])
        self._stds[focal_id] = prediction[0, 2:] * self._args.var_coef

        return _sample_valid_position(focal.get_position(), focal.get_velocity(), prediction, simu.get_timestep(), self._args)


class Multi_plstm_predict_traj:
    def __init__(self, model, num_timesteps, args, num_neighs=1):
        self._model = model
        self._num_timesteps = num_timesteps
        self._num_neighs = num_neighs
        self._selection_method = most_influential_individual[args.most_influential_individual](
            args)
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
            X = np.hstack((X, rad.reshape(-1, 1)))

        ind_idcs = self._selection_method.sort(focal_id, individuals, simu)
        for idx in ind_idcs:
            ind = individuals[idx]
            p2 = ind.get_position_history()
            v2 = ind.get_velocity_history()
            X = np.hstack((X, p2[-self._num_timesteps:, :]))
            X = np.hstack((X, v2[-self._num_timesteps:, :]))
            if self._args.distance_inputs:
                rad = self._compute_dist_wall(p2[-self._num_timesteps:, :])
                X = np.hstack((X, rad.reshape(-1, 1)))

                dist = self._compute_inter_dist(
                    p1[-self._num_timesteps:, :],
                    p2[-self._num_timesteps:, :])
                X = np.hstack((X, dist.reshape(-1, 1)))

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
