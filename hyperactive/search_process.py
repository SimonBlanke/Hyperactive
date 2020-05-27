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


class SearchProcess:
    def __init__(self, nth_process, _main_args_, _info_):
        self.start_time = time.time()
        self.i = 0
        self._main_args_ = _main_args_
        self.memory = _main_args_.memory

        self.memory_dict = {}
        self.memory_dict_new = {}

        self._info_ = _info_()

        self._score = -np.inf
        self._pos = None

        self.score_best = -np.inf
        self.pos_best = None

        self.score_list = []
        self.pos_list = []

        self.nth_process = nth_process
        model_nr = nth_process % _main_args_.n_models
        self.func_ = list(_main_args_.search_config.keys())[model_nr]
        self.search_space = _main_args_.search_config[self.func_]

        self._space_ = SearchSpace(_main_args_, model_nr)
        self.func_name = str(self.func_).split(" ")[1]
        self._model_ = Model(self.func_, nth_process, _main_args_)
        self._init_ = InitSearchPosition(self._space_, self._model_, _main_args_)

        self.eval_time = []
        self.iter_times = []

        if not self.memory:
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        elif self.memory == "short":
            self.mem = None
            self.eval_pos = self.eval_pos_Mem

        elif self.memory == "long":
            self.mem = Hypermemory(
                _main_args_.X, _main_args_.y, self.func_, self.search_space,
            )
            self.eval_pos = self.eval_pos_Mem

            self.memory_dict = self.mem.load()

        else:
            print("Warning: Memory not defined")
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        if self.mem:
            if self.mem.meta_data_found:
                self.pos_best = self.mem.pos_best
                self.score_best = self.mem.score_best

    def _init_eval(self):
        self.pos_best = self._init_._set_start_pos(self._info_)
        self.score_best = self.eval_pos(self.pos_best)

    def _get_warm_start(self):
        return self._space_.pos2para(self.pos_best)

    def _process_results(self):
        self.total_time = time.time() - self.start_time
        start_point = self._info_.print_start_point(self)

        if self._main_args_.memory == "long":
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
        para = self._space_.pos2para(pos)
        para["iteration"] = self.i
        results = self._model_.train_model(para)

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
