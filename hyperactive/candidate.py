# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .search_space import SearchSpace
from .model import Model
from .init_position import InitSearchPosition


class Candidate:
    def __init__(
        self, obj_func, func_para, search_space, init_para, memory, verb, hyperactive
    ):
        self.obj_func = obj_func
        self.func_para = func_para
        self.search_space = search_space
        self.memory = memory
        self.verb = verb

        self.space = SearchSpace(search_space, verb)
        self.model = Model(obj_func, func_para, verb, hyperactive)
        self.init = InitSearchPosition(init_para, self.space, verb)

        self.memory_dict = {}
        self.memory_dict_new = {}

        self._score = -np.inf
        self._pos = None

        self.score_best = -np.inf
        self.pos_best = None

        self.score_list = []
        self.pos_list = []

        self.eval_time = []
        self.iter_times = []

        if not memory:
            self.mem = None
            self.eval_pos = self.eval_pos_noMem
        else:
            self.mem = None
            self.eval_pos = self.eval_pos_Mem

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self.score_list.append(value)
        self._score = value

    @property
    def pos(self):
        return self._score

    @pos.setter
    def pos(self, value):
        self.pos_list.append(value)
        self._pos = value

    def base_eval(self, pos, nth_iter):
        para = self.space.pos2para(pos)
        para["iteration"] = nth_iter
        results = self.model.eval(para)

        if results["score"] > self.score_best:
            self.score_best = results["score"]
            self.pos_best = pos

            self.verb.p_bar.best_since_iter = nth_iter

        return results

    def eval_pos_noMem(self, pos, nth_iter):
        results = self.base_eval(pos, nth_iter)
        return results["score"]

    def eval_pos_Mem(self, pos, nth_iter, force_eval=False):
        pos.astype(int)
        pos_tuple = tuple(pos)

        if pos_tuple in self.memory_dict and not force_eval:
            return self.memory_dict[pos_tuple]["score"]
        else:
            results = self.base_eval(pos, nth_iter)
            self.memory_dict[pos_tuple] = results
            self.memory_dict_new[pos_tuple] = results

            return results["score"]

    def get_score(self, pos_new, nth_iter):
        score_new = self.eval_pos(pos_new, nth_iter)
        self.verb.p_bar.update_p_bar(1, self.score_best)

        if score_new > self.score_best:
            self.score = score_new
            self.pos = pos_new

        return score_new
