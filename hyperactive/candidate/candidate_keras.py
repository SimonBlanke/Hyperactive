# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .candidate import Candidate
from ..model import KerasModel
from ..init_position import InitDLSearchPosition


class KerasCandidate(Candidate):
    def __init__(self, nth_process, _config_):
        super().__init__(nth_process, _config_)

        self.nth_process = nth_process
        self.func_ = list(_config_.search_config.keys())[0]

        self._space_.create_kerasSearchSpace()
        self._model_ = KerasModel(_config_)

        self._init_ = InitDLSearchPosition(
            self._space_, self._model_, _config_.warm_start, _config_.scatter_init
        )

    def create_start_point(self, para):
        start_point = {}

        temp_dict = {}
        for para_key in para:
            temp_dict[para_key] = [para[para_key]]

        start_point[self.func_] = temp_dict

        return start_point

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self.create_start_point(para_best)

        return warm_start
