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

    def create_kerasSearchSpace(self):
        search_space = {}

        for layer_str in self.search_config.keys():

            for param_str in self.search_config[layer_str].keys():
                new_param_str = layer_str + "." + param_str

                search_space[new_param_str] = self.search_config[layer_str][param_str]

        self.para_space = search_space

    def create_mlSearchSpace(self, model_str):
        self.para_space = self.search_config[model_str]

    def warm_start_ml(self, nth_process):
        for key in self.warm_start.keys():
            model_str, start_process = key.rsplit(".", 1)

            if int(start_process) == nth_process:
                hyperpara_indices = self._set_start_position_sklearn(nth_process)
            else:
                hyperpara_indices = self.get_random_position()

        return hyperpara_indices

    def warm_start_dl(self, nth_process):
        for key in self.warm_start.keys():
            model_str, start_process = key.rsplit(".", 1)

            if int(start_process) == nth_process:
                hyperpara_indices = self._set_start_position_keras(nth_process)
            else:
                hyperpara_indices = self.get_random_position()

        return hyperpara_indices

    def _add_list_to_dict_values(self, dict_, model_str):
        dict__ = {}

        for key in self.search_config[model_str].keys():
            dict__[key] = [dict_[key]]

        return dict__

    def set_default_warm_start(self):
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

    def _set_start_position_keras(self, nth_process):
        pos = {}

        for layer_key in self.para_space.keys():
            layer_str, para_str = layer_key.rsplit(".", 1)

            search_position = self.para_space[layer_key].index(
                *self.warm_start[layer_str][para_str]
            )

            pos[layer_key] = search_position

        return pos

    def _get_model(self, model):
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model

    def _set_start_position_sklearn(self, nth_process):
        pos = {}

        for hyperpara_name in self.para_space.keys():
            start_point_key = list(self.warm_start.keys())[nth_process]

            try:
                search_position = self.para_space[hyperpara_name].index(
                    *self.warm_start[start_point_key][hyperpara_name]
                )
            except ValueError:
                print("Warm start not in search space, using random position")
                return self.get_random_position()

            pos[hyperpara_name] = search_position

        return pos

    def get_random_position(self):
        """
        get a random N-Dim position in search space and return:
        N indices of N-Dim position (dict)
        """
        pos = {}

        for hyperpara_name in self.para_space.keys():
            n_hyperpara_values = len(self.para_space[hyperpara_name])
            search_position = random.randint(0, n_hyperpara_values - 1)

            pos[hyperpara_name] = search_position

        return pos

    def pos2para(self, pos_space):
        para = {}

        for hyperpara_name in pos_space.keys():
            pos = pos_space[hyperpara_name]
            para[hyperpara_name] = list(self.para_space[hyperpara_name])[pos]

        return para

    def pos_dict2np_array(self, pos_list):
        pos_np = []

        for pos in pos_list:
            pos_np.append(np.array(list(pos.values())))

        return np.array(pos_np)

    def pos_np2values_dict(self, pos_np):
        pos = []

        for pos_np_ in pos_np:
            values_dict = {}
            for i, key in enumerate(self.para_space.keys()):
                pos_ = int(pos_np_[i])
                values_dict[key] = list(self.para_space[key])[pos_]

            pos.append(values_dict)

        return pos
