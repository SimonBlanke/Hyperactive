# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class HyperGradientTrafo:
    def __init__(self, search_space):
        self.search_space = search_space
        self.paras_n = list(self.search_space.keys())

        self.search_space_values = list(self.search_space.values())

        search_space_positions = {}
        for key in search_space.keys():
            search_space_positions[key] = np.array(range(len(search_space[key])))
        self.search_space_positions = search_space_positions

        self.data_types = {}
        for para in search_space.keys():
            space_dim = np.array(list(self.search_space[para]))
            try:
                np.subtract(space_dim, space_dim)
            except:
                _type_ = "object"
            else:
                _type_ = "number"

            self.data_types[para] = _type_

    def value2position(self, value: list) -> list:
        position = []
        for n, space_dim in enumerate(self.search_space_values):
            pos = np.abs(value[n] - np.array(space_dim)).argmin()
            position.append(int(pos))

        return position

    def value2para(self, value: list) -> dict:
        para = {}
        for key, p_ in zip(self.paras_n, value):
            para[key] = p_

        return para

    def para2value(self, para: dict) -> list:
        value = []
        for para_name in self.paras_n:
            value.append(para[para_name])

        return value

    def position2value(self, position):
        value = []

        for n, space_dim in enumerate(self.search_space_values):
            value.append(space_dim[position[n]])

        return value

    def trafo_para(self, para_hyper):
        para_gfo = {}
        for para in self.paras_n:
            value_hyper = para_hyper[para]
            space_dim = list(self.search_space[para])

            if self.data_types[para] == "number":
                value_gfo = np.abs(value_hyper - np.array(space_dim)).argmin()
            else:
                value_gfo = space_dim.index(value_hyper)

            para_gfo[para] = value_gfo
        return para_gfo

    def trafo_initialize(self, initialize):
        if "warm_start" in list(initialize.keys()):
            warm_start_l = initialize["warm_start"]
            warm_start_gfo = []
            for warm_start in warm_start_l:
                para_gfo = self.trafo_para(warm_start)
                warm_start_gfo.append(para_gfo)

            initialize["warm_start"] = warm_start_gfo

        return initialize

    def get_list_positions(self, list1_values, search_dim):
        list_positions = []

        for value2 in list1_values:
            pos_appended = False
            for value1 in search_dim:
                if value1 == value2:
                    list_positions.append(search_dim.index(value1))
                    pos_appended = True
                    break

            if not pos_appended:
                list_positions.append(None)

        return list_positions

    def trafo_memory_warm_start(self, results):
        if results is None:
            return results

        results.reset_index(inplace=True)

        df_positions_dict = {}
        for para_name in self.paras_n:
            result_dim_values = list(results[para_name].values)
            search_dim = self.search_space[para_name]

            # if self.data_types[para_name] == "function":
            #     result_dim_values = [value.__name__ for value in result_dim_values]

            list1_positions = self.get_list_positions(result_dim_values, search_dim)

            # remove None
            # list1_positions_ = [x for x in list1_positions if x is not None]
            df_positions_dict[para_name] = list1_positions

        results_new = pd.DataFrame(df_positions_dict)

        results_new["score"] = results["score"]
        results_new.dropna(how="any", inplace=True)

        return results_new
