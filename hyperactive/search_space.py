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

    def pos_space_limit(self):
        dim = []
        for pos_key in self.para_space:
            dim.append(len(self.para_space[pos_key]) - 1)

        self.dim = np.array(dim)

    def create_kerasSearchSpace(self):
        search_space = {}

        for layer_str in self.search_config.keys():

            for param_str in self.search_config[layer_str].keys():
                new_param_str = layer_str + "." + param_str

                search_space[new_param_str] = self.search_config[layer_str][param_str]

        self.para_space = search_space
        self.pos_space_limit()

    def create_mlSearchSpace(self, model_str):
        self.para_space = self.search_config[model_str]
        self.pos_space_limit()

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
        pos = []

        for hyperpara_name in self.para_space.keys():
            n_para_values = len(self.para_space[hyperpara_name])
            pos_ = random.randint(0, n_para_values - 1)

            pos.append(pos_)

        return np.array(pos)

    def pos2para(self, pos):
        if len(self.para_space.keys()) == pos.size:
            values_dict = {}
            for i, key in enumerate(self.para_space.keys()):
                pos_ = int(pos[i])
                values_dict[key] = list(self.para_space[key])[pos_]

            return values_dict
        else:
            print("\n para_space", self.para_space)
            print("\n pos", pos)
            raise ValueError("para_space and pos have different size")
