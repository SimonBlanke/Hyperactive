# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer


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
        hyperband_init=False,
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
            hyperband_init,
        )

        self.n_part = n_part
        self.w = w
        self.c_k = c_k
        self.c_s = c_s

    def _find_best_particle(self, cand, part_list):
        for p in part_list:
            if p.score_best > cand.score_best:
                cand.score_best = p.score_best
                cand.pos_best = p.pos_best

    def _init_particles(self, cand):
        part_list = [Particle() for _ in range(self.n_part)]
        for i, p in enumerate(part_list):
            p.nr = i
            p.pos = cand._space_.get_random_position()
            p.pos_best = p.pos
            p.velo = np.zeros(len(cand._space_.para_space))

        return part_list

    def _move_particles(self, cand, part_list):

        for p in part_list:
            A = self.w * p.velo
            B = self.c_k * random.random() * np.subtract(p.pos_best, p.pos)
            C = self.c_s * random.random() * np.subtract(cand.pos_best, p.pos)
            new_velocity = A + B + C

            p.velo = new_velocity
            p.move(cand)

    def _eval_particles(self, cand, part_list, X, y):
        for p in part_list:
            para = cand._space_.pos2para(p.pos)
            p.score, _, _ = cand._model_.train_model(para, X, y)

            if p.score > p.score_best:
                p.score_best = p.score
                p.pos_best = p.pos

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        part_list = self._init_particles(_cand_)
        for i in tqdm.tqdm(**self._tqdm_dict(_cand_)):
            self._eval_particles(_cand_, part_list, X, y)
            self._find_best_particle(_cand_, part_list)
            self._move_particles(_cand_, part_list)

        return _cand_


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
