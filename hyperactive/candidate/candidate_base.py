# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from ..search_space import SearchSpace
from ..model import Model
from ..init_position import InitSearchPosition


class Candidate:
    def __init__(self, obj_func, training_data, search_space, init_para, p_bar):
        self.obj_func = obj_func
        self.search_space = search_space
        self.p_bar = p_bar

        self.space = SearchSpace(search_space)
        self.model = Model(obj_func, training_data)
        self.init = InitSearchPosition(init_para, self.space)

        self.memory_dict = {}
        self.memory_dict_new = {}

        self.score_best = -np.inf
        self.pos_best = None
        self.para_best = None

        self.score_list = []
        self.scores_best_list = []
        self.pos_list = []

        self.eval_times = []
        self.iter_times = []

    def base_eval(self, pos):
        para = self.space.pos2para(pos)
        results = self.model.eval(para)

        self.score_list.append(results["score"])
        self.scores_best_list.append(self.score_best)

        if results["score"] > self.score_best:
            self.score_best = results["score"]
            self.pos_best = pos
            self.para_best = para

        return results

    def get_score(self, pos_new):
        score_new = self.evaluate(pos_new)
        self.p_bar.update_p_bar(1, score_new)

        return score_new
