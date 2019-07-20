# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random

import numpy as np

# from .base import BaseOptimizer
from ..local.hill_climbing_optimizer import HillClimber
from .simulated_annealing import SimulatedAnnealingOptimizer


class ParallelTemperingOptimizer(SimulatedAnnealingOptimizer):
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
        eps=1,
        t_rate=0.98,
        n_neighbours=1,
        system_temps=[0.1, 0.2, 0.01],
        n_swaps=10,
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

        self.eps = eps
        self.t_rate = t_rate
        self.temp = 0.1
        self.n_neighbours = n_neighbours
        self.system_temps = system_temps
        self.n_swaps = n_swaps

        self.n_iter_swap = int(self.n_iter / self.n_swaps)

        self.initializer = self._init_tempering

    def _init_annealers(self, cand):
        self.ann_list = [Annealer(temp=temp) for temp in self.system_temps]
        for ind in self.ann_list:
            ind.pos = cand._space_.get_random_pos()
            ind.pos_best = ind.pos

        return self.ann_list

    def _annealing_systems(self, _cand_):
        for annealer in self.ann_list:
            annealer.pos_curr = self._annealing(_cand_)

    def _find_neighbours(self, _cand_):
        for ann in self.ann_list:
            ann.find_neighbour(_cand_)

    def _swap_pos(self, _cand_):
        for ann1 in self.ann_list:
            for ann2 in self.ann_list:
                rand = random.uniform(0, 1)

                score_diff_norm = (ann1.score - ann2.score) / (ann1.score + ann2.score)
                temp = (1 / ann1.temp) - (1 / ann2.temp)
                p_accept = np.exp(score_diff_norm * temp)

                # print("p_accept", p_accept)

                if p_accept > rand:
                    temp_temp = ann1.temp  # haha!
                    ann1.temp = ann2.temp
                    ann1.temp = temp_temp
                    break

    def _eval_annealers(self, cand, X, y):
        for ann in self.ann_list:
            para = cand._space_.pos2para(ann.pos)
            ann.score, _, _ = cand._model_.train_model(para, X, y)

            if ann.score > ann.score_best:
                ann.score_best = ann.score
                ann.pos_best = ann.pos

    def _find_best_annealer(self, cand):
        for ann in self.ann_list:
            if ann.score_best > cand.score_best:
                cand.score_best = ann.score_best
                cand.pos_best = ann.pos_best

    def _iterate(self, i, _cand_, X, y):
        self._annealer_.find_neighbour(_cand_)
        _cand_.pos = self._annealer_.pos
        _cand_.eval(X, y)

        self._find_neighbours(_cand_)
        self._annealing_systems(_cand_)
        self._eval_annealers(_cand_, X, y)
        self._find_best_annealer(_cand_)

        if self.n_iter_swap != 0 and i % self.n_iter_swap == 0:
            self._swap_pos(_cand_)

        return _cand_

    def _init_tempering(self, _cand_):
        self._annealer_ = Annealer()

        self.pos_curr = _cand_.pos
        self.score_curr = _cand_.score

        self.ann_list = self._init_annealers(_cand_)


class Annealer(HillClimber):
    def __init__(self, eps=1, temp=1):
        super().__init__(eps)

        self.pos = None
        self.pos_best = None

        self.score = -1000
        self.score_best = -1000

        self.temp = temp

    def find_neighbour(self, _cand_):
        super().climb(_cand_)
