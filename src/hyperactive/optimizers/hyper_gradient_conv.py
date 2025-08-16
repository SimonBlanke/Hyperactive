"""hyper_gradient_conv module for Hyperactive optimization."""

# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class HyperGradientConv:
    """HyperGradientConv class."""

    def __init__(self, s_space):
        self.s_space = s_space

    def value2position(self, value: list) -> list:
        """Convert values to positions."""
        return [
            np.abs(v - np.array(space_dim)).argmin()
            for v, space_dim in zip(value, self.s_space.values_l)
        ]

    def value2para(self, value: list) -> dict:
        """Convert values to parameters."""
        return {key: p for key, p in zip(self.s_space.dim_keys, value)}

    def para2value(self, para: dict) -> list:
        """Convert parameters to values."""
        return [para[para_name] for para_name in self.s_space.dim_keys]

    def position2value(self, position):
        """Position2Value function."""
        return [
            space_dim[pos] for pos, space_dim in zip(position, self.s_space.values_l)
        ]

    def para_func2str(self, para):
        """Para Func2Str function."""
        return {
            dim_key: (
                para[dim_key].__name__
                if self.s_space.data_types[dim_key] != "number"
                else para[dim_key]
            )
            for dim_key in self.s_space.dim_keys
        }

    def value_func2str(self, value):
        """Value Func2Str function."""
        try:
            return value.__name__
        except AttributeError:
            return value

    def conv_para(self, para_hyper):
        """Conv Para function."""
        para_gfo = {}
        for para in self.s_space.dim_keys:
            value_hyper = para_hyper[para]
            space_dim = list(self.s_space.func2str[para])

            if self.s_space.data_types[para] == "number":
                value_gfo = np.abs(value_hyper - np.array(space_dim)).argmin()
            else:
                value_hyper = self.value_func2str(value_hyper)

                if value_hyper in space_dim:
                    value_gfo = space_dim.index(value_hyper)
                else:
                    raise ValueError(f"'{value_hyper}' was not found in '{para}'")

            para_gfo[para] = value_gfo
        return para_gfo

    def conv_initialize(self, initialize):
        """Conv Initialize function."""
        if "warm_start" in initialize:
            warm_start_l = initialize["warm_start"]
            warm_start_gfo = [self.conv_para(warm_start) for warm_start in warm_start_l]
            initialize["warm_start"] = warm_start_gfo

        return initialize

    def get_list_positions(self, list1_values, search_dim):
        """Get List Positions function."""
        return [search_dim.index(value2) for value2 in list1_values]

    def values2positions(self, values, search_dim):
        """Values2Positions function."""
        return np.array(search_dim).searchsorted(values)

    def positions2results(self, positions):
        """Positions2Results function."""
        results_dict = {}

        for para_name in self.s_space.dim_keys:
            values_list = self.s_space[para_name]
            pos_ = positions[para_name].values
            values_ = [values_list[idx] for idx in pos_]
            results_dict[para_name] = values_

        results = pd.DataFrame.from_dict(results_dict)

        diff_list = np.setdiff1d(positions.columns, results.columns)
        results[diff_list] = positions[diff_list]

        return results

    def conv_memory_warm_start(self, results):
        """Conv Memory Warm Start function."""
        if results is None:
            return results

        results.reset_index(inplace=True, drop=True)

        df_positions_dict = {}
        for dim_key in self.s_space.dim_keys:
            result_dim_values = list(results[dim_key].values)
            search_dim = self.s_space.func2str[dim_key]

            if self.s_space.data_types[dim_key] == "object":
                result_dim_values = [
                    self.value_func2str(value) for value in result_dim_values
                ]

                list1_positions = self.get_list_positions(result_dim_values, search_dim)
            else:
                list1_positions = self.values2positions(result_dim_values, search_dim)

            df_positions_dict[dim_key] = list1_positions

        results_new = pd.DataFrame(df_positions_dict)

        results_new["score"] = results["score"]
        results_new.dropna(how="any", inplace=True)

        return results_new
