# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .search_space import SearchSpace
from .model import MachineLearner
from .model import DeepLearner


class Candidate:
    def __init__(self, nth_process, search_config, warm_start, metric, cv):
        self.warm_start = warm_start
        self.search_config = search_config

        self.score = 0
        self.score_best = 0

        self.pos = None
        self.pos_best = None

        self._space_ = SearchSpace(warm_start, search_config)

    def set_position(self, pos):
        self.pos = pos

    def move(self, move_func):
        self.pos = move_func(self._space_, self.pos)

    def eval(self, X_train, y_train):
        para = self._space_.pos2para(self.pos)

        self.score, _, self.sklearn_model = self._model_.train_model(
            para, X_train, y_train
        )


class MlCandidate(Candidate):
    def __init__(
        self, nth_process, search_config, warm_start, metric, cv, model_module_str
    ):
        super().__init__(nth_process, search_config, warm_start, metric, cv)

        self.nth_process = nth_process
        self.warm_start = warm_start

        self._space_.create_mlSearchSpace(model_module_str)
        self._model_ = MachineLearner(search_config, metric, cv, model_module_str)

        if self.warm_start:
            self.pos = self._space_.warm_start_ml(nth_process)
        else:
            self.pos = self._space_.get_random_position()

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self._model_.create_start_point(para_best, self.nth_process)

        return warm_start


class DlCandidate(Candidate):
    def __init__(self, nth_process, search_config, warm_start, metric, cv):
        super().__init__(nth_process, search_config, warm_start, metric, cv)

        self.nth_process = nth_process
        self.warm_start = warm_start

        self._space_.create_kerasSearchSpace()
        self._model_ = DeepLearner(search_config, metric, cv)
