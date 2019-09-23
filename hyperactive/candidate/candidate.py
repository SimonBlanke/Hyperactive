# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..search_space import SearchSpace
from ..model import Model
from ..init_position import InitSearchPosition


class Candidate:
    def __init__(self, nth_process, _config_):
        self.search_config = _config_.search_config
        self.memory = _config_.memory

        self._score_best = -1000
        self.pos_best = None

        self.model = None
        self._space_ = SearchSpace(_config_)

        self.nth_process = nth_process
        self.func_ = list(_config_.search_config.keys())[0]

        self.func_name = str(self.func_).split(" ")[1]

        self._space_.create_kerasSearchSpace()
        self._model_ = Model(_config_)

        self._init_ = InitSearchPosition(
            self._space_, self._model_, _config_.warm_start, _config_.scatter_init
        )

    def create_start_point(self, para):
        start_point = {}

        temp_dict = {}
        for para_key in para:
            temp_dict[para_key] = [para[para_key]]

        start_point[self.func_name] = temp_dict

        return start_point

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self.create_start_point(para_best)

        return warm_start

    @property
    def score_best(self):
        return self._score_best

    @score_best.setter
    def score_best(self, value):
        # self.model_best = self.model
        self._score_best = value

    def eval_pos(self, pos, X, y, force_eval=False):
        pos_str = pos.tostring()

        if pos_str in self._space_.memory and self.memory and not force_eval:
            return self._space_.memory[pos_str]
        else:
            para = self._space_.pos2para(pos)
            score = self._model_.train_model(para, X, y)
            self._space_.memory[pos_str] = score

            return score
