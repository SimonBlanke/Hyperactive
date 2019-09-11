# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
from tqdm.auto import tqdm
import scipy
import numpy as np
import pandas as pd
import multiprocessing

from .util import merge_dicts


class Config:
    def __init__(self, *args, **kwargs):
        kwargs_base = {
            "n_iter": 10,
            "optimizer": "RandomSearch",
            "n_jobs": 1,
            "verbosity": 2,
            "warnings": True,
            "random_state": None,
            "warm_start": False,
            "memory": True,
            "scatter_init": False,
            "meta_learn": False,
            "proxy_dataset": False,
            "repulsor": False,
            "get_search_path": False,
        }

        self.search_config = args[0]
        kwargs_base = merge_dicts(kwargs_base, kwargs)
        self._set_general_args(kwargs_base)

        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        self.set_n_jobs()
        self._n_process_range = range(0, int(self.n_jobs))

    def _set_general_args(self, kwargs_base):
        self.n_iter = kwargs_base["n_iter"]
        self.optimizer = kwargs_base["optimizer"]
        self.n_jobs = kwargs_base["n_jobs"]
        self.verbosity = kwargs_base["verbosity"]
        self.warnings = kwargs_base["warnings"]
        self.random_state = kwargs_base["random_state"]
        self.warm_start = kwargs_base["warm_start"]
        self.memory = kwargs_base["memory"]
        self.scatter_init = kwargs_base["scatter_init"]
        self.meta_learn = kwargs_base["meta_learn"]
        self.get_search_path = kwargs_base["get_search_path"]

    def init_p_bar(self, _config_, _cand_):
        if self.verbosity == 2:
            self.p_bar = tqdm(**_config_._tqdm_dict(_cand_))
        else:
            self.p_bar = None

    def update_p_bar(self, n):
        if self.p_bar:
            self.p_bar.update(n)

    def close_p_bar(self):
        if self.p_bar:
            self.p_bar.close()

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
