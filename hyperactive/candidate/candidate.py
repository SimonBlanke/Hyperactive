# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..search_space import SearchSpace


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
