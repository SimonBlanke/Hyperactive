# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class Converter:
    def __init__(self, search_space):
        self.search_space = search_space
        self.para_names = list(self.search_space.keys())

    def value2position(self, value):
        position = []
        for n, space_dim in enumerate(self.search_space_values):
            pos = np.abs(value[n] - space_dim).argmin()
            position.append(pos)

        return np.array(position).astype(int)

    def value2para(self, value):
        para = {}
        for key, p_ in zip(self.para_names, value):
            para[key] = p_

        return para

    def position2value(self, position):
        value = []

        for n, space_dim in enumerate(self.search_space_values):
            value.append(space_dim[position[n]])

        return np.array(value)

    def para2value(self, para):

        value = []
        for para_name in self.para_names:
            value.append(para[para_name])

        return np.array(value)


class HyperGradientTrafo(Converter):
    def __init__(self, search_space):
        super().__init__(search_space)
        self.search_space_values = list(self.search_space.values())

        search_space_positions = {}
        for key in search_space.keys():
            search_space_positions[key] = np.array(
                range(len(search_space[key]))
            )
        self.search_space_positions = search_space_positions

    def trafo_initialize(self, initialize):
        if "warm_start" in list(initialize.keys()):
            warm_start = initialize["warm_start"]
            warm_start_gfo = []
            for warm_start_ in warm_start:
                value = self.trafo.para2value(warm_start_)
                position = self.trafo.value2position(value)
                pos_para = self.trafo.value2para(position)

                warm_start_gfo.append(pos_para)

            initialize["warm_start"] = warm_start_gfo

        return initialize

    def trafo_memory_warm_start(self, results):
        if results is None:
            return results

        df_positions_dict = {}
        for para_name in self.para_names:
            list1_values = list(results[para_name].values)
            list1_positions = [self.search_space[para_name].index(value) for value in list1_values]
            df_positions_dict[para_name] = list1_positions

        results_new = pd.DataFrame(df_positions_dict)
        results_new["score"] = results["score"]

        return results_new