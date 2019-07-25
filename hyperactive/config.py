# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import multiprocessing


class Config:
    def __init__(self, *args, **kwargs):
        self.kwargs_base = {
            "metric": "accuracy",
            "n_jobs": 1,
            "cv": 5,
            "verbosity": 1,
            "random_state": None,
            "warm_start": False,
            "memory": True,
            "scatter_init": False,
        }

        if "search_config" in list(kwargs.keys()):
            self.search_config = kwargs["search_config"]
        else:
            self.search_config = args[0]
        if "n_iter" in list(kwargs.keys()):
            self.n_iter = kwargs["n_iter"]
        else:
            self.n_iter = args[1]

        self.kwargs_base.update(kwargs)

        self.set_n_jobs()

    def set_n_jobs(self):
        """Sets the number of jobs to run in parallel"""
        num_cores = multiprocessing.cpu_count()
        if self.kwargs_base["n_jobs"] == -1 or self.kwargs_base["n_jobs"] > num_cores:
            self.kwargs_base["n_jobs"] = num_cores

    def _is_all_same(self, list):
        """Checks if model names in search_config are consistent"""
        if len(set(list)) == 1:
            return True
        else:
            return False

    def _get_model_str(self):
        model_type_list = []

        for model_type_key in self.search_config.keys():
            if "sklearn" in model_type_key:
                model_type_list.append("sklearn")
            elif "xgboost" in model_type_key:
                model_type_list.append("xgboost")
            elif "keras" in model_type_key:
                model_type_list.append("keras")
            elif "torch" in model_type_key:
                model_type_list.append("torch")
            else:
                raise Exception("\n No valid model string in search_config")

        return model_type_list

    def get_model_type(self):
        """extracts the model type from the search_config (important for search space construction)"""
        model_type_list = self._get_model_str()

        if self._is_all_same(model_type_list):
            self.model_type = model_type_list[0]
        else:
            raise Exception("\n Model strings in search_config keys are inconsistent")
