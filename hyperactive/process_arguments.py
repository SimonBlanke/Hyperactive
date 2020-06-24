# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import random
import numpy as np
import multiprocessing

from .checks import check_kwargs


def stop_warnings():
    # because sklearn warnings are annoying when they appear 100 times
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn


class ProcessArguments:
    def __init__(self, args, kwargs, random_state):
        check_kwargs(kwargs)

        self.kwargs = kwargs
        self._set_default()
        self._add_args2kwargs(args)

        self.function_parameter = self.kwargs["function_parameter"]
        self.search_space = self.kwargs["search_space"]
        self.optimizer = self.kwargs["optimizer"]
        self.random_state = random_state
        self.n_jobs = self.kwargs["n_jobs"]
        self.init_para = self.kwargs["init_para"]

        self.set_n_jobs()

        if isinstance(self.optimizer, dict):
            optimizer = list(self.optimizer.keys())[0]
            self.opt_para = self.optimizer[optimizer]
            self.optimizer = optimizer

            self.n_positions = self._get_n_positions()
            print("n_positions", self.n_positions)
        else:
            self.opt_para = {}
            self.n_positions = self._get_n_positions()

    def _get_n_positions(self):
        n_positions_strings = [
            "n_positions",
            "system_temperatures",
            "n_particles",
            "individuals",
        ]

        n_positions = 1
        for n_pos_name in n_positions_strings:
            if n_pos_name in list(self.opt_para.keys()):
                n_positions = self.opt_para[n_pos_name]
                if n_positions == "system_temperatures":
                    n_positions = len(n_positions)

        return n_positions

    def set_n_jobs(self):
        """Sets the number of jobs to run in parallel"""
        num_cores = multiprocessing.cpu_count()
        if self.n_jobs == -1 or self.n_jobs > num_cores:
            self.n_jobs = num_cores

    def get_process_para(self):
        pass

    def _check_parameter(kwargs):
        pass

    def _add_args2kwargs(self, args):
        for arg in args:
            if callable(arg):
                self.kwargs["objective_function"] = arg
            elif isinstance(arg, dict):
                self.kwargs["search_space"] = arg

    def set_random_seed(self, thread):
        """Sets the random seed separately for each thread (to avoid getting the same results in each thread)"""
        if self.random_state is None:
            self.random_state = np.random.randint(0, high=2 ** 32 - 2)

        random.seed(self.random_state + thread)
        np.random.seed(self.random_state + thread)

    def _set_default(self):
        self.kwargs.setdefault("function_parameter", None)
        self.kwargs.setdefault("memory", None)
        self.kwargs.setdefault("optimizer", "RandomSearch")
        self.kwargs.setdefault("n_iter", 10)
        self.kwargs.setdefault("n_jobs", 1)
        self.kwargs.setdefault("init_para", [])
        self.kwargs.setdefault("distribution", None)
