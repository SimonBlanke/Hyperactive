# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np
import tqdm

from .base import BaseOptimizer
from .base import BaseCandidate


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
        n_part=2,
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

    def _init_particles(self, cand):
        print("cand.pos", cand.pos)

        pos_np = cand._space_.pos_dict2np_array(cand.pos)

        cand.best_pos = cand.pos
        cand.velo = np.zeros(pos_np.shape)

        print("pos_np", pos_np, type(pos_np))
        print("cand.best_pos", cand.best_pos, type(cand.best_pos))
        print("cand.velo", cand.velo, type(cand.velo))

    def _get_dim_SearchSpace(self, cand):
        return len(cand._space_.para_space)

    def _limit_pos(self, cand):
        max_pos_list = []
        for values in list(cand._space_.para_space.values()):
            max_pos_list.append(len(values) - 1)

        self.max_pos_list = np.array(max_pos_list)

    def _move(self, cand):
        np_pos = []
        for pos in cand.pos:
            pos_ = cand._space_.pos_dict2np_array(pos)

            np_pos.append(pos_)

        np_pos = np.array(np_pos)

        print("np_pos", np_pos)

        A = self.w * cand.velo
        B = self.c_k * random.random() * np.subtract(self.best_pos, self.pos)
        C = self.c_s * random.random() * np.subtract(self.g_best_pos, self.pos)
        new_velocity = A + B + C

        self.velo = new_velocity

        self.pos = (self.pos + self.velo).astype(int)

        zeros = np.zeros(len(self.pos))
        self.pos = np.maximum(self.pos, zeros)
        self.pos = np.minimum(self.pos, self.max_pos_list)

    def search(self, nth_process, X, y):
        _cand_ = self._init_population_search(nth_process, X, y, self.n_part)

        self._init_particles(_cand_)
        self._limit_pos(_cand_)

        print("\n")
        print("_cand_.pos", _cand_.pos)
        print("_cand_.velo", _cand_.velo)
        print("\n")

        _cand_.eval(X, y)

        for i in tqdm.tqdm(
            range(self.n_steps),
            # desc=str(self.model_str),
            position=nth_process,
            leave=False,
        ):

            self._move(_cand_)

            if _cand_.score > _cand_.score_best:
                self.score_best = _cand_.score
                self.pos_best = _cand_.best_pos

        start_point = _cand_._get_warm_start()

        return _cand_.pos_best, _cand_.score_best, start_point


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
