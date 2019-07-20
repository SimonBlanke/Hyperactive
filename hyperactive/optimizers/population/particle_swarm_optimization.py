# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from ...base import BaseOptimizer


class ParticleSwarmOptimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        memory=True,
        scatter_init=False,
        n_part=4,
        w=0.5,
        c_k=0.5,
        c_s=0.9,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
            memory,
            scatter_init,
        )

        self.n_part = n_part
        self.w = w
        self.c_k = c_k
        self.c_s = c_s

        self.initializer = self._init_part

    def _find_best_particle(self, cand):
        for p in self.part_list:
            if p.score_best > cand.score_best:
                cand.score_best = p.score_best
                cand.pos_best = p.pos_best

    def _init_particles(self, cand):
        self.part_list = [Particle() for _ in range(self.n_part)]
        for i, p in enumerate(self.part_list):
            p.nr = i
            p.pos = cand._space_.get_random_pos()
            p.pos_best = p.pos
            p.velo = np.zeros(len(cand._space_.para_space))

        return self.part_list

    def _move_particles(self, cand):

        for p in self.part_list:
            A = self.w * p.velo
            B = self.c_k * random.random() * np.subtract(p.pos_best, p.pos)
            C = self.c_s * random.random() * np.subtract(cand.pos_best, p.pos)
            new_velocity = A + B + C

            p.velo = new_velocity
            p.move(cand)

    def _eval_particles(self, cand, X, y):
        for p in self.part_list:
            para = cand._space_.pos2para(p.pos)
            p.score, _, _ = cand._model_.train_model(para, X, y)

            if p.score > p.score_best:
                p.score_best = p.score
                p.pos_best = p.pos

    def _iterate(self, i, _cand_, X, y):
        self._eval_particles(_cand_, X, y)
        self._find_best_particle(_cand_)
        self._move_particles(_cand_)

        if self._show_progress_bar():
            self.p_bar.update(1)

        return _cand_

    def _init_part(self, _cand_):
        self.part_list = self._init_particles(_cand_)


class Particle:
    def __init__(self):
        self.nr = None
        self.pos = None
        self.pos_best = None

        self.score = -1000
        self.score_best = -1000

        self.velo = None

    def move(self, cand):
        self.pos = (self.pos + self.velo).astype(int)
        # limit movement
        n_zeros = [0] * len(cand._space_.dim)
        self.pos = np.clip(self.pos, n_zeros, cand._space_.dim)
