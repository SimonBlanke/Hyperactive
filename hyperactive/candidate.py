# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .search_space import SearchSpace
from .model import MachineLearner
from .model import DeepLearner


class Candidate:
    def __init__(self, nth_process, search_config, warm_start, metric, cv):
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

    def eval(self, X, y):
        para = self._space_.pos2para(self.pos)

        self.score, _, self.sklearn_model = self._model_.train_model(para, X, y)


class MlCandidate(Candidate):
    def __init__(
        self, nth_process, search_config, warm_start, metric, cv, model_module_str
    ):
        super().__init__(nth_process, search_config, warm_start, metric, cv)

        self.nth_process = nth_process

        self._space_.create_mlSearchSpace(model_module_str)
        self._model_ = MachineLearner(search_config, metric, cv, model_module_str)

        if warm_start:
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

        self._space_.create_kerasSearchSpace()
        self._model_ = DeepLearner(search_config, metric, cv)

        if warm_start:
            self.pos = self._space_.warm_start_dl(nth_process)
        else:
            self.pos = self._space_.get_random_position()

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self._model_.trafo_hyperpara_dict_lists(para_best)

        return warm_start


"""
class Candidates:
    def __init__(self, nth_process, search_config, warm_start, metric, cv, n_cand):
        self.search_config = search_config

        self.score = []
        self.score_best = []

        self.pos = []
        self.pos_best = []

        self._space_ = SearchSpace(warm_start, search_config)

    def set_position(self, pos):
        self.pos = pos

    def eval(self, X, y):
        for pos in self.pos:
            para = self._space_.pos2para(pos)
            score, _, _ = self._model_.train_model(para, X, y)

            self.score.append(score)


class MlCandidates(Candidates):
    def __init__(
        self,
        nth_process,
        search_config,
        warm_start,
        metric,
        cv,
        n_cand,
        model_module_str,
    ):
        super().__init__(nth_process, search_config, warm_start, metric, cv, n_cand)

        self._space_.create_mlSearchSpace(model_module_str)
        self._model_ = MachineLearner(search_config, metric, cv, model_module_str)

        for cand in range(n_cand):
            self.pos.append(self._space_.get_random_position())

        if warm_start:
            self.pos.append(self._space_.warm_start_ml(nth_process))

        print("\n init self.pos", self.pos, "\n")


class DlCandidates(Candidates):
    def __init__(self, nth_process, search_config, warm_start, metric, cv, n_cand):
        super().__init__(nth_process, search_config, warm_start, metric, cv, n_cand)

        self._space_.create_kerasSearchSpace()
        self._model_ = DeepLearner(search_config, metric, cv)

        for cand in n_cand:
            self.pos.append(self._space_.get_random_position())

        if warm_start:
            self.pos.append(self._space_.warm_start_dl(nth_process))
"""
