# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import random

from .particle_swarm_optimization import ParticleSwarmOptimizer
from ..local import HillClimbingPositioner


class EvolutionStrategyOptimizer(ParticleSwarmOptimizer):
    def __init__(self, _main_args_, _opt_args_):
        super().__init__(_main_args_, _opt_args_)
        self.n_pop = self._opt_args_.individuals

        self.n_mutations = int(round(self.n_pop * self._opt_args_.mutation_rate))
        self.n_crossovers = int(round(self.n_pop * self._opt_args_.crossover_rate))

    def _init_individuals(self, _cand_):
        _p_list_ = [Individual(**self._opt_args_.kwargs_opt) for _ in range(self.n_pop)]
        for _p_ in _p_list_:
            _p_.pos_new = _p_.move_random(_cand_)

            self._optimizer_eval(_cand_, _p_)
            _p_.pos_current = _p_.pos_new
            _p_.score_current = _p_.score_new

            print("\n_p_.pos_current\n", _p_.pos_current)

        return _p_list_

    def _mutate_individuals(self, _cand_, mutate_idx):
        for _p_ in self._p_list_[mutate_idx]:
            _p_.pos_new = _p_.move_climb(_cand_, _p_.pos_current)

    def _crossover(self, _cand_, cross_idx, replace_idx):
        for i, _p_ in enumerate(self._p_list_[replace_idx]):
            j = i + 1
            if j == len(cross_idx):
                j = 0

            pos_new = self._cross_two_ind(
                [self._p_list_[cross_idx][i], self._p_list_[cross_idx][j]]
            )

            _p_.pos_new = pos_new

    def _cross_two_ind(self, _p_list_):
        pos_new = []

        print("_p_list_[0].pos_current", _p_list_[0].pos_current)
        print("_p_list_[1].pos_current", _p_list_[1].pos_current)

        for pos1, pos2 in zip(_p_list_[0].pos_current, _p_list_[1].pos_current):
            rand = random.randint(0, 1)
            if rand == 0:
                pos_new.append(pos1)
            else:
                pos_new.append(pos2)

        return np.array(pos_new)

    def _move_positioners(self, _cand_):
        idx_sorted_ind = self._rank_individuals()
        mutate_idx, cross_idx, replace_idx = self._select_individuals(idx_sorted_ind)

        self._crossover(_cand_, cross_idx, replace_idx)
        self._mutate_individuals(_cand_, mutate_idx)

    def _rank_individuals(self):
        scores_list = []
        for _p_ in self._p_list_:
            scores_list.append(_p_.score_current)

        scores_np = np.array(scores_list)
        idx_sorted_ind = list(scores_np.argsort()[::-1])

        return idx_sorted_ind

    def _select_individuals(self, index_best):
        mutate_idx = index_best[: self.n_mutations]
        cross_idx = index_best[: self.n_crossovers]

        n = self.n_crossovers
        replace_idx = index_best[-n:]

        return mutate_idx, cross_idx, replace_idx

    def _iterate(self, i, _cand_):
        _p_current = self._p_list_[i % self.n_pop]
        print("\n_p_current.pos_current\n", _p_current.pos_current)
        self._move_positioners(_cand_)
        self._update_pos(_cand_, _p_current)

        return _cand_

    def _init_iteration(self, _cand_):
        _p_list_ = self._init_individuals(_cand_)
        self._p_list_ = np.array(_p_list_)

    def _finish_search(self):
        self._pbar_.close_p_bar()

        return self._p_list_


class Individual(HillClimbingPositioner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
