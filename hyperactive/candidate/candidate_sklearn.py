# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .candidate import Candidate
from ..model import MachineLearner
from ..init_position import InitMLSearchPosition


class MlCandidate(Candidate):
    def __init__(
        self,
        nth_process,
        search_config,
        metric,
        cv,
        warm_start,
        memory,
        scatter_init,
        model_module_str,
    ):
        super().__init__(
            nth_process, search_config, metric, cv, warm_start, memory, scatter_init
        )

        self.nth_process = nth_process

        self._space_.create_mlSearchSpace(model_module_str)
        self._model_ = MachineLearner(search_config, metric, cv, model_module_str)

        self._init_ = InitMLSearchPosition(
            self._space_, self._model_, warm_start, scatter_init
        )

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self._model_.create_start_point(para_best, self.nth_process)

        return warm_start
