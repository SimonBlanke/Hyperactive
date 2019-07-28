# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .candidate import Candidate
from ..model import MachineLearner
from ..init_position import InitMLSearchPosition


class MlCandidate(Candidate):
    def __init__(self, nth_process, _config_):
        super().__init__(nth_process, _config_)

        self.nth_process = nth_process
        search_config_key = _config_._get_sklearn_model(nth_process)

        self._space_.create_mlSearchSpace(search_config_key)
        self._model_ = MachineLearner(_config_, search_config_key)

        self._init_ = InitMLSearchPosition(
            self._space_, self._model_, _config_.warm_start, _config_.scatter_init
        )

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self._model_.create_start_point(para_best, self.nth_process)

        return warm_start
