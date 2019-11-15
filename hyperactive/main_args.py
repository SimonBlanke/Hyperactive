# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np
import multiprocessing

from .util import merge_dicts


class MainArgs:
    def __init__(self, *args, **kwargs):
        self.X, self.y = args[0], args[1]

        self.args_default_class = {
            "verbosity": 2,
            "random_state": None,
            "memory": "long",
            "get_search_path": False,
        }
        self.args_default_method = {
            "n_iter": 100,
            "max_time": None,
            "optimizer": "RandomSearch",
            "n_jobs": 1,
            "warm_start": False,
            "scatter_init": False,
        }

        # self.search_config = args[0]
        self.opt_para = dict()
        self.args_class = merge_dicts(self.args_default_class, kwargs)

        self.verbosity = self.args_class["verbosity"]
        self.random_state = self.args_class["random_state"]
        self.memory = self.args_class["memory"]
        self.get_search_path = self.args_class["get_search_path"]

    def search_args(self, *args, **kwargs):
        self.search_config = args[0]
        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        self.args_method = merge_dicts(self.args_default_method, kwargs)

        self.n_iter = self.args_method["n_iter"]
        self.max_time = self.args_method["max_time"]
        self.optimizer = self.args_method["optimizer"]
        self.n_jobs = self.args_method["n_jobs"]
        self.warm_start = self.args_method["warm_start"]
        self.scatter_init = self.args_method["scatter_init"]

        if self.max_time:
            self.max_time = self.max_time * 3600

        self.set_n_jobs()

        self._n_process_range = range(0, int(self.n_jobs))

        if "optimizer" in kwargs and isinstance(kwargs["optimizer"], dict):
            self.optimizer = list(kwargs["optimizer"].keys())[0]
            self.opt_para = kwargs["optimizer"][self.optimizer]

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
