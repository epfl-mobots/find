import os
import numpy as np
import warnings
import tqdm
import socket
import datetime

from random import shuffle

class Simulation:
    def __init__(self, timestep, num_iterations, args={'stats_enabled': False, 'simu_dir_gen': True}):
        self._individual_list = []
        self._descriptor_list = []
        self._stat_list = []
        self._num_iterations = num_iterations
        self._current_iteration = 0
        self._args = args
        self._timestep = timestep
        self._dirname = ''

        if 'simu_dir_gen' in self._args.keys() and self._args['simu_dir_gen']:
            hostname = socket.gethostname()
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self._dirname = hostname + '_' + timestamp
            if not os.path.exists(self._dirname):
                os.makedirs(self._dirname)

    def get_simu(self):
        return self

    def add_individual(self, individual):
        self._individual_list.append(individual)
        return self

    def add_descriptor(self, desc):
        self._descriptor_list.append(desc)
        return self

    def add_stat(self, stat):
        self._stat_list.append(stat)
        return self

    def get_num_individuals(self):
        return len(self._individual_list)

    def get_individuals(self):
        return self._individual_list

    def get_num_iterations(self):
        return self._num_iterations

    def get_current_iteration(self):
        return self._current_iteration

    def get_dirname(self):
        return self._dirname

    def get_descriptors(self):
        return self._descriptor_list

    def get_stats(self):
        return self._stat_list

    def get_timestep(self):
        return self._timestep

    def _update(self):
        if 'stats_enabled' in self._args.keys() and self._args['stats_enabled']:
            for obj in self._stat_list:
                obj(self)

            for obj in self._descriptor_list:
                obj(self)

    def _dump(self):
        if 'stats_enabled' in self._args.keys() and self._args['stats_enabled']:
            for obj in self._stat_list:
                obj.save()

            for obj in self._descriptor_list:
                obj.save()

    def spin_once(self):
        ind_ids = list(range(len(self._individual_list)))
        shuffle(ind_ids)
        for idx in ind_ids:
            self._individual_list[idx].run(self)
        
        if 'stats_enabled' in self._args.keys() and self._args['stats_enabled']:
            self._update()

        if self._current_iteration > self._num_iterations:
            warnings.warn(
                'You have exceeded the number of iterations allocated for this simulation')

        self._current_iteration += 1

    def spin(self):
        for _ in tqdm.tqdm(range(self._num_iterations)):
            self.spin_once()

        if 'stats_enabled' in self._args.keys() and self._args['stats_enabled']:
            self._dump()
