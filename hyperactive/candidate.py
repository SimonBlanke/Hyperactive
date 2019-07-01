# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .search_space import SearchSpace
from .model import MachineLearner
from .model import DeepLearner


class Candidate:
    def __init__(self, nth_process, search_config, metric, cv, warm_start, memory):
        self.search_config = search_config
        self.memory = memory

        self.score = -1000
        self._score_best = -1000

        self.pos = None
        self.pos_best = None

        self._space_ = SearchSpace(search_config, warm_start, memory)

    @property
    def score_best(self):
        return self._score_best

    @score_best.setter
    def score_best(self, value):
        self.model_best = self.model_trained
        self._score_best = value

    def eval(self, X, y):
        para = self._space_.pos2para(self.pos)

        pos = self.pos.tostring()
        if pos in self._space_.memory and self.memory:
            self.score = self._space_.memory[pos]
        else:

            self.score, _, self.model_trained = self._model_.train_model(para, X, y)

            self._space_.memory[pos] = self.score


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

    def _get_finished_model(self, warm_start):
        model, nr = list(warm_start.keys())[0].rsplit(".", 1)

        model = self._model_._get_model(model)
        para = warm_start[list(warm_start.keys())[0]]

        # convert listed values to unlisted values
        sklearn_para = {}
        for para_key in para:
            sklearn_para[para_key] = para[para_key][0]

        sklearn_model = model(**sklearn_para)

        return sklearn_model


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
