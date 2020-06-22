# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class InitSearchPosition:
    def __init__(self, init_para, space, verb):
        self.init_para = init_para
        self.space = space
        self.verb = verb

    def set_start_pos(self, n_inits):
        positions = []
        for init in self.init_para:
            self.verb.info.warm_start()
            pos = self._warm_start_one(init)
            positions.append(pos)

        for init in range(len(self.init_para), n_inits):
            self.verb.info.random_start()
            pos = self.space.get_random_pos()
            positions.append(pos)

        return positions

    def _warm_start_one(self, init_para):
        pos = []

        init_para_names = list(init_para.keys())
        for hyperpara_name in self.space.search_space.keys():
            if hyperpara_name not in init_para_names:
                search_position = self.space.get_random_pos_scalar(hyperpara_name)

            else:
                search_position = self.space.search_space[hyperpara_name].index(
                    init_para[hyperpara_name]
                )
            pos.append(search_position)

        return np.array(pos)
