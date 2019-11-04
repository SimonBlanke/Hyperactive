# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import random
import numpy as np


class SearchSpace:
    def __init__(self, _core_, model_nr):
        self.search_config = _core_.search_config
        self.warm_start = _core_.warm_start
        self.scatter_init = _core_.scatter_init
        self.model_nr = model_nr

        self.memory = {}

    def load_memory(self, para, score):
        if para is None or score is None:
            return
        for idx in range(para.shape[0]):
            pos = self.para2pos(para.iloc[[idx]])
            pos_str = pos.tostring()
            self.memory[pos_str] = float(score.values[idx])

    def pos_space_limit(self):
        dim = []

        for pos_key in self.para_space:
            dim.append(len(self.para_space[pos_key]) - 1)

        self.dim = np.array(dim)

    def create_searchspace(self):
        self.para_space = self.search_config[list(self.search_config)[self.model_nr]]
        self.pos_space_limit()

    def get_random_pos(self):
        pos_new = np.random.uniform(np.zeros(self.dim.shape), self.dim, self.dim.shape)
        pos = np.rint(pos_new).astype(int)

        return pos

    def get_random_pos_scalar(self, hyperpara_name):
        n_para_values = len(self.para_space[hyperpara_name])
        pos = random.randint(0, n_para_values - 1)

        return pos

    def para2pos(self, para):
        pos_list = []

        for pos_key in self.para_space:
            value = para[[pos_key]].values

            pos = self.para_space[pos_key].index(value)
            pos_list.append(pos)

        return np.array(pos_list)

    def pos2para(self, pos):
        if len(self.para_space.keys()) == pos.size:
            values_dict = {}
            for i, key in enumerate(self.para_space.keys()):
                pos_ = int(pos[i])
                values_dict[key] = list(self.para_space[key])[pos_]

            return values_dict
