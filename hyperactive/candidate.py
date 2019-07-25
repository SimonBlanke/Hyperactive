# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .search_space import SearchSpace
from .model import MachineLearner
from .model import DeepLearner

from .init_position import InitMLSearchPosition
from .init_position import InitDLSearchPosition


class Candidate:
    def __init__(
        self, nth_process, search_config, metric, cv, warm_start, memory, scatter_init
    ):
        self.search_config = search_config
        self.memory = memory

        self._score_best = -1000
        self.pos_best = None

        self._space_ = SearchSpace(search_config, warm_start, scatter_init)

    def eval_pos(self, pos, X, y):
        pos_str = pos.tostring()

        if pos_str in self._space_.memory and self.memory:
            return self._space_.memory[pos_str]

        else:
            para = self._space_.pos2para(pos)
            score, _, _ = self._model_.train_model(para, X, y)

            self._space_.memory[pos_str] = score

            return score


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


class DlCandidate(Candidate):
    def __init__(
        self, nth_process, search_config, metric, cv, warm_start, memory, scatter_init
    ):
        super().__init__(
            nth_process, search_config, metric, cv, warm_start, memory, scatter_init
        )

        self.nth_process = nth_process

        self._space_.create_kerasSearchSpace()
        self._model_ = DeepLearner(search_config, metric, cv)

        self._init_ = InitDLSearchPosition(
            self._space_, self._model_, warm_start, scatter_init
        )

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self._model_.trafo_hyperpara_dict_lists(para_best)

        return warm_start
