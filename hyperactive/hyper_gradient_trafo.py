# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numbers
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

    def positions2values(self, positions):
        values_temp = []
        positions_np = np.array(positions)

        for n, space_dim in enumerate(self.search_space_values):
            pos_1d = positions_np[:, n]
            value_ = np.take(space_dim, pos_1d, axis=0)
            values_temp.append(value_)

        values = list(np.array(values_temp).T)
        return values

    def para2value(self, para):
        value = []
        for para_name in self.para_names:
            value.append(para[para_name])

        return np.array(value)

    def _memory2dataframe(self, memory_dict):
        positions = [np.array(pos).astype(int) for pos in list(memory_dict.keys())]
        scores = list(memory_dict.values())

        memory_positions = pd.DataFrame(positions, columns=self.para_names)
        memory_positions["score"] = scores

        return memory_positions


class HyperGradientTrafo(Converter):
    def __init__(self, search_space):
        super().__init__(search_space)
        self.search_space_values = list(self.search_space.values())

        search_space_positions = {}
        for key in search_space.keys():
            search_space_positions[key] = np.array(range(len(search_space[key])))
        self.search_space_positions = search_space_positions

        """
        self.search_space_ltm = {}
        self.data_types = {}
        for para_name in search_space.keys():
            value0 = search_space[para_name][0]

            if isinstance(value0, numbers.Number):
                type0 = "number"
                search_dim_ltm = search_space[para_name]
            elif isinstance(value0, str):
                type0 = "string"
                search_dim_ltm = search_space[para_name]

            elif callable(value0):
                type0 = "function"

                search_dim_ltm = []
                for func in list(search_space[para_name]):
                    search_dim_ltm.append(func.__name__)

            else:
                type0 = None
                search_dim_ltm = search_space[para_name]

            self.data_types[para_name] = type0
            self.search_space_ltm[para_name] = search_dim_ltm
        """

    def trafo_initialize(self, initialize):
        if "warm_start" in list(initialize.keys()):
            warm_start = initialize["warm_start"]
            warm_start_gfo = []
            for warm_start_ in warm_start:
                value = self.para2value(warm_start_)
                position = self.value2position(value)
                pos_para = self.value2para(position)

                warm_start_gfo.append(pos_para)

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

        df_positions_dict = {}
        for para_name in self.para_names:
            result_dim_values = list(results[para_name].values)
            search_dim = self.search_space[para_name]

            # if self.data_types[para_name] == "function":
            #     result_dim_values = [value.__name__ for value in result_dim_values]

            # print("\n para_name", para_name)

            # print(" result_dim_values", result_dim_values)
            # print(" search_dim", search_dim)

            list1_positions = self.get_list_positions(result_dim_values, search_dim)

            # remove None
            # list1_positions_ = [x for x in list1_positions if x is not None]
            df_positions_dict[para_name] = list1_positions

        results_new = pd.DataFrame(df_positions_dict)
        results_new["score"] = results["score"]
        results_new.dropna(how="any", inplace=True)

        return results_new
