# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

# from importlib import import_module


class InitSearchPosition:
    def __init__(self, space, model, warm_start, hyperband_init):
        self._space_ = space
        self._model_ = model
        self.warm_start = warm_start
        self.hyperband_init = hyperband_init

    def _set_start_pos(self, nth_process, X, y):
        if self.warm_start and self.hyperband_init:
            if len(list(self.warm_start.keys())) > nth_process:
                pos = self._create_warm_start(nth_process)
            else:
                pos = self._hyperband_init(nth_process, X, y)

        elif self.warm_start:
            if len(list(self.warm_start.keys())) > nth_process:
                pos = self._create_warm_start(nth_process)
            else:
                pos = self._space_.get_random_pos()

        elif self.hyperband_init:
            pos = self._hyperband_init(nth_process, X, y)

        else:
            pos = self._space_.get_random_pos()

        return pos

    def _hyperband_init(self, nth_process, X, y):

        pos_list = []
        for i in range(self.hyperband_init):
            pos = self._space_.get_random_pos()
            pos_list.append(pos)

        """
        print("\n test:")
        for pos in pos_list:
            para = self._space_.pos2para(pos)
            score, _, _ = self._model_.train_model(para, X, y)

            print("score", score)
            print("pos  ", pos)

        print("\n\n")
        """

        hb_init = self.hyperband_init
        while hb_init > 1:
            pos_best_list, score_best_list = self._hyperband_train(
                X, y, hb_init, pos_list
            )

            pos_best_sorted, score_best_sorted = self._sort_for_best(
                pos_best_list, score_best_list
            )

            hb_init = int(hb_init / 2)
            pos_list = pos_best_sorted[:hb_init]

        return pos_list[0]

    def _sort_for_best(self, sort, sort_by):
        sort = np.array(sort)
        sort_by = np.array(sort_by)

        index_best = list(sort_by.argsort()[::-1])

        sort_sorted = sort[index_best]
        sort_by_sorted = sort_by[index_best]

        return sort_sorted, sort_by_sorted

    def _hyperband_train(self, X, y, hb_init, pos_list):
        X_list = np.array_split(X, hb_init)
        y_list = np.array_split(y, hb_init)

        pos_best_list = []
        score_best_list = []
        for X_, y_, pos in zip(X_list, y_list, pos_list):
            para = self._space_.pos2para(pos)
            score, _, _ = self._model_.train_model(para, X_, y_)

            pos_best_list.append(pos)
            score_best_list.append(score)

        return pos_best_list, score_best_list


class InitMLSearchPosition(InitSearchPosition):
    def __init__(self, space, model, warm_start, hyperband_init):
        super().__init__(space, model, warm_start, hyperband_init)

    def _create_warm_start(self, nth_process):
        pos = []

        for hyperpara_name in self._space_.para_space.keys():
            start_point_key = list(self.warm_start.keys())[nth_process]

            try:
                search_position = self._space_.para_space[hyperpara_name].index(
                    *self.warm_start[start_point_key][hyperpara_name]
                )
            except ValueError:
                print("Warm start not in search space, using random position")
                return self._space_.get_random_pos()

            pos.append(search_position)

        return np.array(pos)

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
    def __init__(self, space, model, warm_start, hyperband_init):
        super().__init__(space, model, warm_start, hyperband_init)

    def _create_warm_start(self, nth_process):
        pos = []

        for layer_key in self._space_.para_space.keys():
            layer_str, para_str = layer_key.rsplit(".", 1)

            search_position = self._space_.para_space[layer_key].index(
                *self.warm_start[layer_str][para_str]
            )

            pos.append(search_position)

        return np.array(pos)
