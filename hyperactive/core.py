# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np
import multiprocessing

from .util import merge_dicts


class Core:
    def __init__(self, *args, **kwargs):
        kwargs_base = {
            "n_iter": 10,
            "max_time": None,
            "optimizer": "RandomSearch",
            "n_jobs": 1,
            "verbosity": 2,
            "random_state": None,
            "warm_start": False,
            "memory": "short",
            "scatter_init": False,
            "get_search_path": False,
        }

        self.search_config = args[0]
        self.opt_para = dict()

        if "optimizer" in kwargs and isinstance(kwargs["optimizer"], dict):
            opt = list(kwargs["optimizer"].keys())[0]
            self.opt_para = kwargs["optimizer"][opt]

            kwargs["optimizer"] = opt

        kwargs_base = merge_dicts(kwargs_base, kwargs)
        self._set_general_args(kwargs_base)

        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        self.set_n_jobs()
        self._n_process_range = range(0, int(self.n_jobs))

        if self.max_time:
            self.max_time = self.max_time * 3600

    def _set_general_args(self, kwargs_base):
        self.n_iter = kwargs_base["n_iter"]
        self.max_time = kwargs_base["max_time"]
        self.optimizer = kwargs_base["optimizer"]
        self.n_jobs = kwargs_base["n_jobs"]
        self.verbosity = kwargs_base["verbosity"]
        self.random_state = kwargs_base["random_state"]
        self.warm_start = kwargs_base["warm_start"]
        self.memory = kwargs_base["memory"]
        self.scatter_init = kwargs_base["scatter_init"]
        self.get_search_path = kwargs_base["get_search_path"]

    def _set_random_seed(self, thread=0):
        """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
        if self.random_state:
            rand = int(self.random_state)
        else:
            rand = 0

        random.seed(rand + thread)
        np.random.seed(rand + thread)

    def set_n_jobs(self):
        """Sets the number of jobs to run in parallel"""
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores
