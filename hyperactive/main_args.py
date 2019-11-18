# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np
import multiprocessing


class MainArgs:
    def __init__(self, X, y, verbosity, random_state, memory):
        self.X = X
        self.y = y
        self.verbosity = verbosity
        self.random_state = random_state
        self.memory = memory
        self.get_search_path = False

        if verbosity > 9:
            self.get_search_path = True

        if self.verbosity > 2:
            self.verbosity = 2

        self.opt_para = dict()

    def search_args(
        self,
        search_config,
        max_time,
        n_iter,
        optimizer,
        n_jobs,
        warm_start,
        scatter_init,
    ):
        self.search_config = search_config
        self.max_time = max_time
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.warm_start = warm_start
        self.scatter_init = scatter_init

        self.model_list = list(self.search_config.keys())
        self.n_models = len(self.model_list)

        if self.max_time:
            self.max_time = self.max_time * 3600

        self.set_n_jobs()

        self._n_process_range = range(0, int(self.n_jobs))

        if isinstance(optimizer, dict):
            self.optimizer = list(optimizer.keys())[0]
            self.opt_para = optimizer[self.optimizer]

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
