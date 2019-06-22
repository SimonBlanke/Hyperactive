# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer
from .base import BaseCandidate
from .search_space import SearchSpace


class ParticleSwarm_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        n_part=1,
        w=0.5,
        c_k=0.5,
        c_s=0.9,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
        )

        self.n_part = n_part
        self.w = w
        self.c_k = c_k
        self.c_s = c_s

        self.best_score = 0
        self.best_pos = None

        self.search_space_inst = SearchSpace(warm_start, search_config)

    def _init_particles(self):
        p_list = [
            Particle(self.model, self.w, self.c_k, self.c_s) for _ in range(self.n_part)
        ]
        for p in p_list:
            p.max_pos_list = self.max_pos_list
            p.pos = self.search_space_inst.pos_dict2np_array(
                self.search_space_inst.get_random_position()
            )
            p.best_pos = self.search_space_inst.pos_dict2np_array(
                self.search_space_inst.get_random_position()
            )
            p.velo = np.zeros(self._get_dim_SearchSpace())

        return p_list

    def _get_dim_SearchSpace(self):
        return len(self.search_space_inst.search_space)

    def _limit_pos(self, search_space):
        max_pos_list = []
        for values in list(search_space.values()):
            max_pos_list.append(len(values) - 1)

        self.max_pos_list = np.array(max_pos_list)

    def _search(self, n_process, X_train, y_train):
        hyperpara_indices = self._init_search(n_process, X_train, y_train)
        self._set_random_seed(n_process)
        self.n_steps = self._set_n_steps(n_process)
        self._limit_pos(self.search_space_inst.search_space)

        self.particle_list = self._init_particles()

        hyperpara_dict = self.search_space_inst.pos_dict2values_dict(hyperpara_indices)
        self.best_pos = self.search_space_inst.pos_dict2np_array(hyperpara_indices)

        for particle in self.particle_list:
            particle.set_position(hyperpara_dict)
            particle.eval(X_train, y_train)

        for particle in self.particle_list:
            if particle.score > self.best_score:
                self.best_score = particle.score
                self.best_pos = particle.best_pos

        if self.metric_type == "score":
            return self._search_best_score(n_process, X_train, y_train)
        elif self.metric_type == "loss":
            return self._search_best_loss(n_process, X_train, y_train)

    def _search_best_score(self, n_process, X_train, y_train):
        for i in tqdm.tqdm(
            range(self.n_steps),
            desc=str(self.model_str),
            position=n_process,
            leave=False,
        ):

            for particle in self.particle_list:
                hyperpara_dict = self.search_space_inst.pos_np2values_dict(particle.pos)
                particle.set_position(hyperpara_dict)
                particle.eval(X_train, y_train)

            for particle in self.particle_list:
                if particle.score > self.best_score:
                    self.best_score = particle.score
                    self.best_pos = particle.best_pos

            for particle in self.particle_list:
                particle.move(self.best_pos)

        best_hyperpara_dict = self.search_space_inst.pos_np2values_dict(self.best_pos)
        score_best, train_time, sklearn_model = self.model.train_model(
            best_hyperpara_dict, X_train, y_train
        )

        start_point = self._finish_search(best_hyperpara_dict, n_process)

        return sklearn_model, score_best, start_point

    def _search_best_loss(self, n_process, X_train, y_train):
        for i in tqdm.tqdm(
            range(self.n_steps),
            desc=str(self.model_str),
            position=n_process,
            leave=False,
        ):

            for particle in self.particle_list:
                hyperpara_dict = self.search_space_inst.pos_np2values_dict(particle.pos)
                particle.set_position(hyperpara_dict)
                particle.eval(X_train, y_train)

            for particle in self.particle_list:
                if particle.score < self.best_score:
                    self.best_score = particle.score
                    self.best_pos = particle.best_pos

            for particle in self.particle_list:
                particle.move(self.best_pos)

        best_hyperpara_dict = self.search_space_inst.pos_np2values_dict(self.best_pos)
        score_best, train_time, sklearn_model = self.model.train_model(
            best_hyperpara_dict, X_train, y_train
        )

        start_point = self._finish_search(best_hyperpara_dict, n_process)

        return sklearn_model, score_best, start_point


class Particle(BaseCandidate):
    def __init__(self, model, w, c_k, c_s):
        super().__init__(model)
        self.w = w
        self.c_k = c_k
        self.c_s = c_s

        self.pos = None
        self.velo = None
        self.score = None

        self.best_pos = None
        self.best_score = 0

        self.sklearn_model = None

        self.max_pos_list = None

    def move(self, best_pos):
        A = self.w * self.velo
        B = self.c_k * random.random() * np.subtract(self.best_pos, self.pos)
        C = self.c_s * random.random() * np.subtract(best_pos, self.pos)
        new_velocity = A + B + C

        self.velo = new_velocity

        self.pos = (self.pos + self.velo).astype(int)

        zeros = np.zeros(len(self.pos))
        self.pos = np.maximum(self.pos, zeros)
        self.pos = np.minimum(self.pos, self.max_pos_list)
