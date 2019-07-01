# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


# from importlib import import_module


class InitSearchPosition:
    def __init__(self, para_space, warm_start, hyperband_init):
        self.para_space = para_space
        self.warm_start = warm_start
        self.hyperband_init = hyperband_init


class InitMLSearchPosition(InitSearchPosition):
    def __init__(self, para_space, warm_start, hyperband_init):
        super().__init__(para_space, warm_start, hyperband_init)

    def warm_start_ml(self, nth_process):
        for key in self.warm_start.keys():
            model_str, start_process = key.rsplit(".", 1)

            if int(start_process) == nth_process:
                hyperpara_indices = self._set_start_position_sklearn(nth_process)
            else:
                hyperpara_indices = self.get_random_position()

        return hyperpara_indices

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

    """
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


    def _get_model(self, model):
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model
    """


class InitDLSearchPosition(InitSearchPosition):
    def __init__(self, para_space, warm_start, hyperband_init):
        super().__init__(para_space, warm_start, hyperband_init)

    def warm_start_dl(self, nth_process):
        for key in self.warm_start.keys():
            model_str, start_process = key.rsplit(".", 1)

            if int(start_process) == nth_process:
                hyperpara_indices = self._set_start_position_keras(nth_process)
            else:
                hyperpara_indices = self.get_random_position()

        return hyperpara_indices

    def _set_start_position_keras(self, nth_process):
        pos = {}

        for layer_key in self.para_space.keys():
            layer_str, para_str = layer_key.rsplit(".", 1)

            search_position = self.para_space[layer_key].index(
                *self.warm_start[layer_str][para_str]
            )

            pos[layer_key] = search_position

        return pos
