# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import scipy
import numpy as np
import pandas as pd
import multiprocessing

from .util import merge_dicts


class Config:
    def __init__(self, *args, **kwargs):
        kwargs_base = {
            "metric": "accuracy_score",
            "n_jobs": 1,
            "cv": 0.75,
            "verbosity": 1,
            "random_state": None,
            "warm_start": False,
            "memory": True,
            "scatter_init": False,
        }

        self.pos_args = ["search_config", "n_iter"]

        self._process_pos_args(args, kwargs)
        kwargs_base = merge_dicts(kwargs_base, kwargs)
        self._set_general_args(kwargs_base)

        self._get_model_str()
        if "metric" not in kwargs:
            self._set_default_metric()

        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        self.set_n_jobs()
        self._n_process_range = range(0, int(self.n_jobs))

    def _process_pos_args(self, args, kwargs):
        pos_args_attr = [None, None]

        for idx, pos_arg in enumerate(self.pos_args):
            if pos_arg in list(kwargs.keys()):
                pos_args_attr[idx] = kwargs[pos_arg]
            else:
                pos_args_attr[idx] = args[idx]

        self.search_config = pos_args_attr[0]
        self.n_iter = pos_args_attr[1]

    def _set_general_args(self, kwargs_base):
        self.metric = kwargs_base["metric"]
        self.n_jobs = kwargs_base["n_jobs"]
        self.cv = kwargs_base["cv"]
        self.verbosity = kwargs_base["verbosity"]
        self.random_state = kwargs_base["random_state"]
        self.warm_start = kwargs_base["warm_start"]
        self.memory = kwargs_base["memory"]
        self.scatter_init = kwargs_base["scatter_init"]

    def _is_all_same(self, list):
        same = False
        """Checks if model names in search_config are consistent"""
        if len(set(list)) == 1:
            same = True

        return same

    def _set_default_metric(self):
        default_metrics = {
            "sklearn": "accuracy_score",
            "xgboost": "accuracy_score",
            "lightgbm": "accuracy_score",
            "catboost": "accuracy_score",
            "keras": "mean_squared_error",
            "torch": "mean_squared_error",
        }

        self.metric = default_metrics[self.model_type]

    def _get_model_str(self):
        model_keys = list(self.search_config.keys())

        module_str_list = []
        for model_key in model_keys:
            module_str = model_key.split(".")[0]
            module_str_list.append(module_str)

        if self._is_all_same(module_str_list):
            self.model_type = model_keys[0].split(".")[0]

    def _tqdm_dict(self, _cand_):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""
        return {
            "total": self.n_iter,
            "desc": "Search " + str(_cand_.nth_process),
            "position": _cand_.nth_process,
            "leave": False,
        }

    def _set_random_seed(self, thread=0):
        """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
        if self.random_state:
            rand = int(self.random_state)
        else:
            rand = 0

        random.seed(rand + thread)
        np.random.seed(rand + thread)
        scipy.random.seed(rand + thread)

    def _get_sklearn_model(self, nth_process):
        """Gets a model_key from the model_list for each thread"""
        if self.n_models > self.n_jobs:
            diff = self.n_models - self.n_jobs

            if nth_process == 0:
                print(
                    "\nNot enough jobs to process models. The last",
                    diff,
                    "model(s) will not be processed",
                )
            model_key = self.model_list[nth_process]
        elif nth_process < self.n_models:
            model_key = self.model_list[nth_process]
        else:
            model_key = random.choice(self.model_list)

        return model_key

    def _show_progress_bar(self):
        show = False

        if self.model_type == "keras" or self.model_type == "torch":
            return show

        show = True

        return show

    def _check_data(self, X, y):
        """Checks if data is pandas Dataframe and converts to numpy array if necessary"""
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(y, pd.core.frame.DataFrame):
            y = y.values

        return X, y

    def set_n_jobs(self):
        """Sets the number of jobs to run in parallel"""
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores
