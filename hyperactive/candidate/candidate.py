# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..search_space import SearchSpace
from ..model import Model
from ..init_position import InitSearchPosition


class Candidate:
    def __init__(self, nth_process, _main_args_):
        self.i = 0
        self.memory = _main_args_.memory

        self._score_best = -1000
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

    def _get_warm_start(self):
        return self._space_.pos2para(self.pos_best)

    @property
    def score_best(self):
        return self._score_best

    @score_best.setter
    def score_best(self, value):
        self.model_best = self.model
        self._score_best = value

    def eval_pos(self, pos, force_eval=False):
        pos_str = pos.tostring()

        if pos_str in self._space_.memory and self.memory and not force_eval:
            return self._space_.memory[pos_str]
        else:
            para = self._space_.pos2para(pos)
            para["iteration"] = self.i
            score, eval_time, self.model = self._model_.train_model(para)
            self._space_.memory[pos_str] = score
            self.eval_time_sum = self.eval_time_sum + eval_time

            return score
