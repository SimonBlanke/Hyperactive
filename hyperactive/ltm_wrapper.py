# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
from hyperactive_long_term_memory import LongTermMemory as _LongTermMemory_


class LongTermMemory:
    def __init__(
        self,
        model_id,
        experiment_id="default",
        save_on="finish",
    ):

        path, _ = os.path.realpath(__file__).rsplit("/", 1)
        path = path + "/"

        self.ltm_origin = _LongTermMemory_(
            model_id=model_id, experiment_id=experiment_id, path="."
        )
        self.save_on = save_on

        if save_on == "finish":
            self.ltm_obj_func_wrapper = self._no_ltm_wrapper
        elif save_on == "iteration":
            self.ltm_obj_func_wrapper = self._ltm_wrapper

    def _no_ltm_wrapper(self, results, para):
        pass

    def _ltm_wrapper(self, results, para):
        if isinstance(results, tuple):
            score = results[0]
            results_dict = results[1]
        else:
            score = results
            results_dict = {}

        results_dict["score"] = score
        ltm_dict = {**para, **results_dict}
        self.save_on_iteration(ltm_dict)

    def clean_files(self):
        self.ltm_origin.clean_files()

    def init_data_types(self, search_space):
        self.ltm_origin.init_data_types(search_space)

    def load(self):
        return self.ltm_origin.load()

    def save_on_finish(self, dataframe):
        self.ltm_origin.save_on_finish(dataframe)

    def save_on_iteration(self, data_dict):
        self.ltm_origin.save_on_iteration(data_dict)
