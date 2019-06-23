# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np

from importlib import import_module


class SearchSpace:
    def __init__(self, warm_start, search_config):
        self.warm_start = warm_start
        self.search_config = search_config

    def create_kerasSearchSpace(self, search_config):
        search_space = {}

        for layer_str in search_config.keys():

            for param_str in search_config[layer_str].keys():
                new_param_str = layer_str + "." + param_str

                search_space[new_param_str] = search_config[layer_str][param_str]

        self.search_space = search_space

    def create_mlSearchSpace(self, search_config, model_str):
        self.search_space = search_config[model_str]

    def warm_start_ml(self, n_process):
        for key in self.warm_start.keys():
            model_str, start_process = key.rsplit(".", 1)

            if int(start_process) == n_process:
                hyperpara_indices = self._set_start_position_sklearn(n_process)
            else:
                hyperpara_indices = self.get_random_position()

        return hyperpara_indices

    def warm_start_dl(self, n_process):
        for key in self.warm_start.keys():
            model_str, start_process = key.rsplit(".", 1)

            if int(start_process) == n_process:
                hyperpara_indices = self._set_start_position_keras(n_process)
            else:
                hyperpara_indices = self.get_random_position()

        return hyperpara_indices

    def _add_list_to_dict_values(self, dict_, model_str):
        dict__ = {}

        for key in self.search_config[model_str].keys():
            dict__[key] = [dict_[key]]

        return dict__

    def set_warm_start(self):
        if self.warm_start is False:
            warm_start = {}
            for i, model_str in enumerate(self.search_config.keys()):
                model = self._get_model(model_str)
                warm_start_dict = self._add_list_to_dict_values(
                    model().get_params(), model_str
                )

                dict_key = model_str + "." + str(i)
                warm_start[dict_key] = warm_start_dict

            self.warm_start = warm_start

    def _set_start_position_keras(self, n_process):
        pos_dict = {}

        for layer_key in self.search_space.keys():
            layer_str, para_str = layer_key.rsplit(".", 1)

            search_position = self.search_space[layer_key].index(
                *self.warm_start[layer_str][para_str]
            )

            pos_dict[layer_key] = search_position

        return pos_dict

    def _get_model(self, model):
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model

    def _set_start_position_sklearn(self, n_process):
        pos_dict = {}

        for hyperpara_name in self.search_space.keys():
            start_point_key = list(self.warm_start.keys())[n_process]

            try:
                search_position = self.search_space[hyperpara_name].index(
                    *self.warm_start[start_point_key][hyperpara_name]
                )
            except ValueError:
                print("Warm start not in search space, using random position")
                return self.get_random_position()

            pos_dict[hyperpara_name] = search_position

        return pos_dict

    def get_random_position(self):
        """
        get a random N-Dim position in search space and return:
        N indices of N-Dim position (dict)
        """
        pos_dict = {}

        for hyperpara_name in self.search_space.keys():
            n_hyperpara_values = len(self.search_space[hyperpara_name])
            search_position = random.randint(0, n_hyperpara_values - 1)

            pos_dict[hyperpara_name] = search_position

        return pos_dict

    def pos_dict2values_dict(self, pos_dict):
        values_dict = {}

        for hyperpara_name in pos_dict.keys():
            pos = pos_dict[hyperpara_name]
            values_dict[hyperpara_name] = list(self.search_space[hyperpara_name])[pos]

        return values_dict

    def pos_dict2np_array(self, pos_dict):
        return np.array(list(pos_dict.values()))

    def pos_np2values_dict(self, np_array):
        if len(self.search_space.keys()) == np_array.size:
            values_dict = {}
            for i, key in enumerate(self.search_space.keys()):
                pos = int(np_array[i])
                values_dict[key] = list(self.search_space[key])[pos]

            return values_dict
        else:
            raise ValueError("search_space and np_array have different size")
