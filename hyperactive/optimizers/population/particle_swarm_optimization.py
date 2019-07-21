# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

from ...base import BaseOptimizer
from ...base import BasePositioner


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
        n_part=10,
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

    def _init_particles(self, _cand_):
        _p_list_ = [Particle() for _ in range(self.n_part)]
        for i, _p_ in enumerate(_p_list_):
            _p_.nr = i
            _p_.pos_current = _cand_._space_.get_random_pos()
            _p_.pos_best = _p_.pos_current
            _p_.velo = np.zeros(len(_cand_._space_.para_space))

        return _p_list_

    def _move_particles(self, _cand_, _p_list_):
        for _p_ in _p_list_:
            r1, r2 = random.random(), random.random()

            A = self.w * _p_.velo
            B = self.c_k * r1 * np.subtract(_p_.pos_best, _p_.pos_current)
            C = self.c_s * r2 * np.subtract(_cand_.pos_best, _p_.pos_current)

            new_velocity = A + B + C

            _p_.velo = new_velocity
            _p_.pos_new = _p_.move_part(_cand_, _p_.pos_current)

    def _eval_particles(self, _cand_, _p_list_, X, y):
        for _p_ in _p_list_:
            _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

            if _p_.score_new > _cand_.score_best:
                _cand_.score_best = _p_.score_new
                _cand_.pos_best = _p_.pos_new

                _p_.pos_current = _p_.pos_new
                _p_.score_current = _p_.score_new

    def _iterate(self, i, _cand_, _p_list_, X, y):
        self._move_particles(_cand_, _p_list_)
        self._eval_particles(_cand_, _p_list_, X, y)

        return _cand_

    def _init_part(self, _cand_, X, y):
        _p_list_ = self._init_particles(_cand_)

        return _p_list_


class Particle(BasePositioner):
    def __init__(self):
        self.nr = None
        self.velo = None

    def move_part(self, _cand_, pos):
        pos_new = (pos + self.velo).astype(int)
        # limit movement
        n_zeros = [0] * len(_cand_._space_.dim)
        return np.clip(pos_new, n_zeros, _cand_._space_.dim)
