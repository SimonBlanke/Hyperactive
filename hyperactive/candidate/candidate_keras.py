# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .candidate import Candidate
from ..model import DeepLearner
from ..init_position import InitDLSearchPosition


class DlCandidate(Candidate):
    def __init__(self, nth_process, _config_):
        super().__init__(nth_process, _config_)

        self.nth_process = nth_process

        self._space_.create_kerasSearchSpace()
        self._model_ = DeepLearner(_config_)

        self._init_ = InitDLSearchPosition(
            self._space_, self._model_, _config_.warm_start, _config_.scatter_init
        )

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self._model_.trafo_hyperpara_dict_lists(para_best)

        return warm_start
