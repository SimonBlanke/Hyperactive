# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer
from .search_space import SearchSpace
from .model import MachineLearner
from .model import DeepLearner


class ParticleSwarm_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        scoring="accuracy",
        tabu_memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
        n_part=1,
        w=0.5,
        c_k=0.5,
        c_s=0.9,
    ):
        super().__init__(
            search_config,
            n_iter,
            scoring,
            tabu_memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            start_points,
        )

        self.n_part = n_part
        self.w = w
        self.c_k = c_k
        self.c_s = c_s

        self.best_score = 0
        self.best_pos = None

        self.particle_search_space = SearchSpace(start_points, search_config)

        if self.model_type == "sklearn":
            self.particle_search_space.create_mlSearchSpace(search_config)
            self.model = MachineLearner(search_config, scoring, cv)
        elif self.model_type == "keras":
            self.particle_search_space.create_kerasSearchSpace(search_config)
            self.model = DeepLearner(search_config, scoring, cv)

        self._limit_pos(self.particle_search_space.search_space)

    def _find_best_particle(self, p_list):
        for p in p_list:
            if p.best_score > self.best_score:
                self.best_score = p.best_score
                self.best_pos = p.best_pos

    def _init_particles(self):
        p_list = [Particle() for _ in range(self.n_part)]
        for p in p_list:
            p.max_pos_list = self.max_pos_list
            p.pos = self.particle_search_space.pos_dict2np_array(
                self.particle_search_space.get_random_position()
            )
            p.best_pos = self.particle_search_space.pos_dict2np_array(
                self.particle_search_space.get_random_position()
            )
            p.velo = np.zeros(self._get_dim_SearchSpace())

        return p_list

    def _get_dim_SearchSpace(self):
        return len(self.particle_search_space.search_space)

    def _limit_pos(self, search_space):
        max_pos_list = []
        for values in list(search_space.values()):
            max_pos_list.append(len(values) - 1)

        self.max_pos_list = np.array(max_pos_list)

    def _move_particles(self, p_list):

        for p in p_list:
            A = self.w * p.velo
            B = self.c_k * random.random() * np.subtract(p.best_pos, p.pos)
            C = self.c_s * random.random() * np.subtract(self.best_pos, p.pos)
            new_velocity = A + B + C

            p.velo = new_velocity
            p.move()

    def _eval_particles(self, p_list, X_train, y_train):
        for p in p_list:
            hyperpara_dict = self.particle_search_space.pos_np2values_dict(p.pos)
            p.score, _, p.sklearn_model = self.model.train_model(
                hyperpara_dict, X_train, y_train
            )

            if p.score > p.best_score:
                p.best_score = p.score
                p.best_pos = p.pos

    def _search(self, n_process, X_train, y_train):
        self._set_random_seed(n_process)
        n_steps = self._set_n_steps(n_process)

        hyperpara_indices = self.particle_search_space.init_eval(n_process)
        hyperpara_dict = self.particle_search_space.pos_dict2values_dict(
            hyperpara_indices
        )
        self.best_pos = self.particle_search_space.pos_dict2np_array(hyperpara_indices)
        self.best_score, train_time, sklearn_model = self.model.train_model(
            hyperpara_dict, X_train, y_train
        )

        p_list = self._init_particles()
        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):
            self._eval_particles(p_list, X_train, y_train)
            self._find_best_particle(p_list)
            self._move_particles(p_list)

        hyperpara_dict_best = self.particle_search_space.pos_np2values_dict(
            self.best_pos
        )
        score_best, train_time, sklearn_model = self.model.train_model(
            hyperpara_dict_best, X_train, y_train
        )

        return sklearn_model, score_best, hyperpara_dict_best, train_time


class Particle:
    def __init__(self):
        self.pos = None
        self.velo = None
        self.score = None

        self.best_pos = None
        self.best_score = 0

        self.sklearn_model = None

        self.max_pos_list = None

    def move(self):
        self.pos = (self.pos + self.velo).astype(int)

        zeros = np.zeros(len(self.pos))
        self.pos = np.maximum(self.pos, zeros)
        self.pos = np.minimum(self.pos, self.max_pos_list)
