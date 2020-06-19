# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import time
import numpy as np

from .search_space import SearchSpace
from .model import Model
from .init_position import InitSearchPosition

from hypermemory import Hypermemory


def meta_data_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/meta_data/"


class ShortTermMemory:
    def __init__(self, _space_, _main_args_, _cand_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = {}
        self.memory_dict_new = {}

        self.meta_data_found = False

        self.n_dims = None


class HypermemoryWrapper:
    def __init__(self):
        pass

    def load_memory(self, X, y):
        self.mem = Hypermemory(X, y, self.obj_func, self.search_space,)
        self.eval_pos = self.eval_pos_Mem
        self.memory_dict = self.mem.load()


class SearchProcess:
    def __init__(
        self,
        nth_process,
        study_para,
        obj_func,
        search_space,
        opt_class,
        n_iter,
        n_jobs,
        init,
        distribution,
        _pbar_,
        _info_,
    ):
        self.study_para = study_para
        self.obj_func = obj_func
        self.search_space = search_space
        self.opt_class = opt_class
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.init = init
        self.distribution = distribution

        self._pbar_ = _pbar_
        self._info_ = _info_()
        self._pbar_.init_p_bar(nth_process, n_iter, obj_func)

        self.start_time = time.time()
        self.i = 0
        # self._main_args_ = _main_args_
        # self.memory = _main_args_.memory

        self.memory = None

        self.memory_dict = {}
        self.memory_dict_new = {}

        self._score = -np.inf
        self._pos = None

        self.score_best = -np.inf
        self.pos_best = None

        self.score_list = []
        self.pos_list = []

        self.space = SearchSpace(search_space, init)
        self.model = Model(obj_func, study_para["obj_func_para"])
        # self._init_ = InitSearchPosition(self._space_, self._model_, _main_args_)

        self.eval_time = []
        self.iter_times = []

        self._memory_processor()

    def _memory_processor(self):
        if not self.memory:
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        elif self.memory == "short":
            self.mem = None
            self.eval_pos = self.eval_pos_Mem

        elif self.memory == "long":
            self._load_memory()

        else:
            print("Warning: Memory not defined")
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        if self.mem:
            if self.mem.meta_data_found:
                self.pos_best = self.mem.pos_best
                self.score_best = self.mem.score_best

    def _get_warm_start(self):
        return self.space.pos2para(self.pos_best)

    def _process_results(self):
        self.total_time = time.time() - self.start_time
        start_point = self._info_.print_start_point(self)

        if self.study_para["memory"] == "long":
            self.mem.dump(self.memory_dict_new)

        return start_point

    def init_pos(self, n_positions):
        init_pos_list = []

        for i in range(n_positions):
            init_pos = self._init_._set_start_pos(self._info_)
            init_pos_list.append(init_pos)

        return init_pos_list

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

    def base_eval(self, pos, p_bar, nth_iter):
        para = self.space.pos2para(pos)
        para["iteration"] = self.i
        results = self.model.eval(para)

        if results["score"] > self.score_best:
            self.score_best = results["score"]
            self.pos_best = pos

            p_bar.best_since_iter = nth_iter

        return results

    def eval_pos_noMem(self, pos, p_bar, nth_iter):
        results = self.base_eval(pos, p_bar, nth_iter)
        return results["score"]

    def eval_pos_Mem(self, pos, p_bar, nth_iter, force_eval=False):
        pos.astype(int)
        pos_tuple = tuple(pos)

        if pos_tuple in self.memory_dict and not force_eval:
            return self.memory_dict[pos_tuple]["score"]
        else:
            results = self.base_eval(pos, p_bar, nth_iter)
            self.memory_dict[pos_tuple] = results
            self.memory_dict_new[pos_tuple] = results

            return results["score"]

    def _get_score(self, pos_new, nth_iter):
        score_new = self.eval_pos(pos_new, self._pbar_, nth_iter)
        self._pbar_.update_p_bar(1, self)

        if score_new > self.score_best:
            self.score = score_new
            self.pos = pos_new

        return score_new

    def search(self, nth_process):
        # self._initialize_search(self._main_args_, nth_process, self._info_)
        """
        if "n_positions" in self._main_args_.opt_para:
            n_positions = self._main_args_.opt_para["n_positions"]
        else:
            n_positions = 1

        init_positions = self.init_pos(n_positions)
        """
        init_positions = [np.array([1, 1])]
        opt_para = dict()

        self.opt = self.opt_class(init_positions, self.space.dim, opt_para)

        # loop to initialize N positions
        for nth_init in range(len(init_positions)):
            pos_new = self.opt.init_pos(nth_init)
            score_new = self._get_score(pos_new, 0)
            self.opt.evaluate(score_new)

        # loop to do the iterations
        for nth_iter in range(len(init_positions), self.n_iter):
            pos_new = self.opt.iterate(nth_iter)
            score_new = self._get_score(pos_new, nth_iter)
            self.opt.evaluate(score_new)

        self._pbar_.close_p_bar()

        return self, self.opt.p_list
