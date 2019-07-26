# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .init_position import InitSearchPosition


class InitMLSearchPosition(InitSearchPosition):
    def __init__(self, space, model, warm_start, scatter_init):
        super().__init__(space, model, warm_start, scatter_init)

    def _create_warm_start(self, nth_process):
        pos = []

        for hyperpara_name in self._space_.para_space.keys():
            start_point_key = list(self.warm_start.keys())[nth_process]

            if hyperpara_name not in list(self.warm_start[start_point_key].keys()):
                # print(hyperpara_name, "not in warm_start selecting random scalar")
                search_position = self._space_.get_random_pos_scalar(hyperpara_name)

            else:
                search_position = self._space_.para_space[hyperpara_name].index(
                    *self.warm_start[start_point_key][hyperpara_name]
                )

            # what if warm start not in search_config range?

            pos.append(search_position)

        return np.array(pos)
