# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..search_space import SearchSpace
from ..model import Model
from ..init_position import InitSearchPosition
from ..extentions import ShortTermMemory, LongTermMemory


class Candidate:
    def __init__(self, nth_process, _main_args_):
        self.i = 0
        self.memory = _main_args_.memory

        self._score_best = -np.inf
        self.pos_best = None
        self.model = None

        self.nth_process = nth_process
        model_nr = nth_process % _main_args_.n_models
        self.func_ = list(_main_args_.search_config.keys())[model_nr]
        self._space_ = SearchSpace(_main_args_, model_nr)
        self.func_name = str(self.func_).split(" ")[1]
        self._model_ = Model(self.func_, nth_process, _main_args_)
        self._init_ = InitSearchPosition(self._space_, self._model_, _main_args_)

        self.eval_time_sum = 0

        if not self.memory:
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        elif self.memory == "short":
            self.mem = ShortTermMemory(self._space_, _main_args_)
            self.eval_pos = self.eval_pos_Mem

        elif self.memory == "long":
            self.mem = LongTermMemory(self._space_, _main_args_)
            self.eval_pos = self.eval_pos_Mem

            self.mem.load_memory(self.func_)

        else:
            print("Warning: Memory not defined")
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        if self.mem:
            if self.mem.meta_data_found:
                self.pos_best = self.mem.pos_best
                self.score_best = self.mem.score_best

        self.pos_best = self._init_._set_start_pos()
        self.score_best = self.eval_pos(self.pos_best)

    def _get_warm_start(self):
        return self._space_.pos2para(self.pos_best)

    @property
    def score_best(self):
        return self._score_best

    @score_best.setter
    def score_best(self, value):
        self.model_best = self.model
        self._score_best = value

    def eval_pos_noMem(self, pos):
        para = self._space_.pos2para(pos)
        para["iteration"] = self.i
        score, eval_time, self.model = self._model_.train_model(para)
        self.eval_time_sum = self.eval_time_sum + eval_time

        return score

    def eval_pos_Mem(self, pos, force_eval=False):
        pos_str = pos.tostring()

        if pos_str in self.mem.memory_dict and not force_eval:
            return self.mem.memory_dict[pos_str]
        else:
            para = self._space_.pos2para(pos)
            para["iteration"] = self.i
            score, eval_time, self.model = self._model_.train_model(para)
            self.mem.memory_dict[pos_str] = score
            self.eval_time_sum = self.eval_time_sum + eval_time

            return score
