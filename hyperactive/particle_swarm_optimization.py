# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer


class ParticleSwarm_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_space,
        n_iter,
        scoring="accuracy",
        tabu_memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        start_points=None,
        n_part=2,
        w=0.5,
        c_k=0.5,
        c_s=0.9,
    ):
        super().__init__(
            search_space,
            n_iter,
            scoring,
            tabu_memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            start_points,
        )
        self._search = self._start_particle_swarm_optimization

        self.n_part = n_part
        self.w = w
        self.c_k = c_k
        self.c_s = c_s

        self.best_score = 0
        self.best_pos = None

    def _find_best_particle(self, p_list):
        for p in p_list:
            if p.best_score > self.best_score:
                self.best_score = p.best_score
                self.best_pos = p.best_pos

    def _init_particles(self):
        p_list = [Particle() for _ in range(self.n_part)]
        for p in p_list:
            p.max_pos_list = self.max_pos_list
            p.pos = self._pos_dict2np_array(self._get_random_position())
            p.best_pos = self._pos_dict2np_array(self._get_random_position())
            p.velo = np.zeros(self._get_dim_SearchSpace())

        return p_list

    def _move_particles(self, p_list):

        for p in p_list:
            A = self.w * p.velo
            B = self.c_k * random.random() * np.subtract(p.best_pos, p.pos)
            C = self.c_s * random.random() * np.subtract(self.best_pos, p.pos)
            new_velocity = A + B + C

            p.velo = new_velocity
            p.move()

    def _eval_particles(self, p_list):
        for p in p_list:
            hyperpara_dict = self._pos_np2values_dict(p.pos)
            p.score, _, p.sklearn_model = self._train_model(hyperpara_dict)

            if p.score > p.best_score:
                p.best_score = p.score
                p.best_pos = p.pos

    def _start_particle_swarm_optimization(self, n_process):
        self._set_random_seed(n_process)
        n_steps = max(1, int(self.n_iter / self.n_jobs))

        hyperpara_indices = self._init_eval(n_process)
        hyperpara_dict = self._pos_dict2values_dict(hyperpara_indices)
        self.best_pos = self._pos_dict2np_array(hyperpara_indices)
        self.best_score, train_time, sklearn_model = self._train_model(hyperpara_dict)

        p_list = self._init_particles()
        for i in tqdm.tqdm(range(n_steps), position=n_process, leave=False):
            self._eval_particles(p_list)
            self._find_best_particle(p_list)
            self._move_particles(p_list)

        hyperpara_dict_best = self._pos_np2values_dict(self.best_pos)
        score_best, train_time, sklearn_model = self._train_model(hyperpara_dict_best)

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
