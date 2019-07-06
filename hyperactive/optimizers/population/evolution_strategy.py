# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
import numpy as np
import random

from ...base import BaseOptimizer

# from .hill_climbing_optimizer import HillClimber


class EvolutionStrategyOptimizer(BaseOptimizer):
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
        individuals=10,
        mutation_rate=0.7,
        crossover_rate=0.3,
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

        self.individuals = individuals
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        self.n_mutations = int(round(self.individuals * mutation_rate))
        self.n_crossovers = int(round(self.individuals * crossover_rate))

    def _init_individuals(self, cand):
        ind_list = [Individual() for _ in range(self.individuals)]
        for ind in ind_list:
            ind.pos = cand._space_.get_random_pos()
            ind.pos_best = ind.pos

        return ind_list

    def _mutate_individuals(self, cand, ind_list, mutate_idx):
        ind_list = np.array(ind_list)
        for ind in ind_list[mutate_idx]:
            ind.mutate(cand)

    def _crossover(self, cand, ind_list, cross_idx, replace_idx):
        ind_list = np.array(ind_list)
        for i, ind in enumerate(ind_list[replace_idx]):
            j = i + 1
            if j == len(cross_idx):
                j = 0

            pos_new = self._cross_two_ind(
                [ind_list[cross_idx][i], ind_list[cross_idx][j]]
            )

            ind.pos = pos_new

    def _cross_two_ind(self, ind_list):
        pos_new = []

        for pos1, pos2 in zip(ind_list[0].pos, ind_list[1].pos):
            rand = random.randint(0, 1)
            if rand == 0:
                pos_new.append(pos1)
            else:
                pos_new.append(pos2)

        return np.array(pos_new)

    def _new_generation(self, cand, ind_list):

        idx_sorted_ind = self._rank_individuals(ind_list)
        mutate_idx, cross_idx, replace_idx = self._select_individuals(idx_sorted_ind)

        self._mutate_individuals(cand, ind_list, mutate_idx)
        self._crossover(cand, ind_list, cross_idx, replace_idx)

    def _eval_individuals(self, cand, ind_list, X, y):
        for ind in ind_list:
            para = cand._space_.pos2para(ind.pos)
            ind.score, _, _ = cand._model_.train_model(para, X, y)

            if ind.score > ind.score_best:
                ind.score_best = ind.score
                ind.pos_best = ind.pos

    def _rank_individuals(self, ind_list):
        scores_list = []
        for ind in ind_list:
            scores_list.append(ind.score)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        return idx_sorted_ind

    def _select_individuals(self, index_best):
        mutate_idx = index_best[: self.n_mutations]
        cross_idx = index_best[: self.n_crossovers]

        n = self.individuals - max(self.n_mutations, self.n_crossovers)
        replace_idx = index_best[-n:]

        return mutate_idx, cross_idx, replace_idx

    def _find_best_individual(self, cand, ind_list):
        for ind in ind_list:
            if ind.score_best > cand.score_best:
                cand.score_best = ind.score_best
                cand.pos_best = ind.pos_best

    def search(self, nth_process, X, y):
        _cand_ = self._init_search(nth_process, X, y)

        _cand_.eval(X, y)

        _cand_.score_best = _cand_.score
        _cand_.pos_best = _cand_.pos

        ind_list = self._init_individuals(_cand_)
        for i in tqdm.tqdm(**self._tqdm_dict(_cand_)):

            self._new_generation(_cand_, ind_list)
            self._eval_individuals(_cand_, ind_list, X, y)
            self._find_best_individual(_cand_, ind_list)

        return _cand_


class Individual:
    def __init__(self):
        self.pos = None
        self.pos_best = None

        self.score = -1000
        self.score_best = -1000

    def mutate(self, cand):
        sigma = cand._space_.dim / 10
        pos_new = np.random.normal(self.pos, sigma, self.pos.shape)
        pos_new_int = np.rint(pos_new)

        n_zeros = [0] * len(cand._space_.dim)
        self.pos = np.clip(pos_new_int, n_zeros, cand._space_.dim)
