# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np
import multiprocessing

from .checks import check_hyperactive_para, check_search_para


def stop_warnings():
    # because sklearn warnings are annoying when they appear 100 times
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn


class MainArgs:
    def __init__(self, X, y, memory, random_state, verbosity, warnings, ext_warnings):
        check_hyperactive_para(X, y, memory, random_state, verbosity)

        if not ext_warnings:
            stop_warnings()

        self._verb_ = None
        self.hyperactive_para = {
            "memory": memory,
            "random_state": random_state,
            "verbosity": verbosity,
        }

        self.X = X
        self.y = y
        self.verbosity = verbosity
        self.random_state = random_state
        self.memory = memory

        self.opt_para = dict()

    def search_args(
        self, search_config, max_time, n_iter, optimizer, n_jobs, scheduler, init_config
    ):
        check_search_para(
            search_config, max_time, n_iter, optimizer, n_jobs, scheduler, init_config
        )

        self.search_para = {
            "search_config": search_config,
            "max_time": max_time,
            "n_iter": n_iter,
            "optimizer": optimizer,
            "n_jobs": n_jobs,
            "scheduler": scheduler,
            "init_config": init_config,
        }

        self.search_config = search_config
        self.max_time = max_time
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.scheduler = scheduler
        self.init_config = init_config

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
        if self.n_jobs > 1 and not self.random_state:
            rand = np.random.randint(0, high=2 ** 32 - 2)
            random.seed(rand + thread)
            np.random.seed(rand + thread)

        elif self.random_state:
            rand = int(self.random_state)

            random.seed(rand + thread)
            np.random.seed(rand + thread)

    def set_n_jobs(self):
        """Sets the number of jobs to run in parallel"""
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores
