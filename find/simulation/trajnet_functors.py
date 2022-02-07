import numpy as np

from find.simulation.tf_nn_functors import CircularCorridor, _sample_valid_position, closest_individual, shuffled_individuals, most_influential_individual, get_most_influential_individual

import torch
from copy import deepcopy


class Trajnet_dir:
    def __init__(self, model, num_timesteps, args, num_neighs=1):
        self._lstm_pred = model
        self._model = self._lstm_pred.model
        self._num_timesteps = num_timesteps
        self._num_neighs = num_neighs
        self._selection = most_influential_individual[args.most_influential_individual]
        self._args = args
        # ! this is also very specific to the inputs we use and should be generalized
        self._cc = CircularCorridor(1.0, (0, 0))
        self._radius = 0.25
        self._offset = 1.0
        self._full_pred = [None, None]
        self._normals = [None, None]

    def get_full_pred(self):
        return self._full_pred

    def get_normals(self):
        return self._normals

    def __call__(self, focal_id, simu):
        individuals = simu.get_individuals()
        focal = list(filter(lambda x: x.get_id() == focal_id, individuals))[0]

        X = np.empty((0, 2))
        xy_f = focal.get_position_history()[-self._num_timesteps:, :]
        xy_f = xy_f + self._offset
        ind_idcs = self._selection(focal_id, individuals)

        for i in range(self._num_timesteps):
            X = np.vstack((X, xy_f[i, :]))
            for idx in ind_idcs[:self._num_neighs]:
                ind = individuals[idx]
                xy_n = ind.get_position_history(
                )[-(self._num_timesteps - i), :]
                xy_n = xy_n + self._offset
                X = np.vstack((X, xy_n))
        X = X.reshape(self._num_timesteps, self._num_neighs + 1, 2)

        xy = torch.Tensor(X)
        scene_goal = np.zeros(shape=(2, 2))
        scene_goal = torch.Tensor(scene_goal)
        batch_split = [0, X.shape[1]]
        batch_split = torch.Tensor(batch_split).long()

        modes = 1
        n_predict = 3
        multimodal_outputs = {}

        max_retries = 100
        retries = 0

        while True:
            normals, output_scenes = self._model(
                xy, scene_goal, batch_split, n_predict=n_predict)
            output_scenes = output_scenes.detach().numpy()
            output_primary = output_scenes[-n_predict:, 0]
            output_neighs = output_scenes[-n_predict:, 1:]
            multimodal_outputs[0] = [output_primary, output_neighs]

            prediction = deepcopy(multimodal_outputs[0][0][0])
            prediction = prediction - self._offset

            full_pred = deepcopy(multimodal_outputs[0])
            self._full_pred[focal_id] = full_pred[0] - self._offset
            normals = normals.detach().numpy()

            normals = normals[-n_predict:
                              ].reshape(n_predict, -1)
            self._normals[focal_id] = normals[:, 2:4]

            # keep sampling until there is a valid prediction
            if self._cc.is_valid(self._cc.radius(prediction)):
                break
            else:
                if retries > max_retries:
                    prediction = focal.get_position()
                    break
                else:
                    retries += 1

        return prediction
