# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..search_space import SearchSpace


class Candidate:
    def __init__(self, nth_process, _config_):
        self.search_config = _config_.search_config
        self.memory = _config_.memory

        self._score_best = -1000
        self.pos_best = None

        self._space_ = SearchSpace(_config_)

    def eval_pos(self, pos, X, y):
        pos_str = pos.tostring()

        if pos_str in self._space_.memory and self.memory:
            return self._space_.memory[pos_str]

        else:
            para = self._space_.pos2para(pos)
            score, _, _ = self._model_.train_model(para, X, y)

            self._space_.memory[pos_str] = score

            return score
