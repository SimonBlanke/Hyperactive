# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from .init_position import InitSearchPosition


class InitDLSearchPosition(InitSearchPosition):
    def __init__(self, space, model, warm_start, scatter_init):
        super().__init__(space, model, warm_start, scatter_init)

    def _create_warm_start(self, nth_process):
        pos = []

        for layer_key in self._space_.para_space.keys():
            layer_str, para_str = layer_key.rsplit(".", 1)

            search_position = self._space_.para_space[layer_key].index(
                *self.warm_start[layer_str][para_str]
            )

            pos.append(search_position)

        return np.array(pos)
