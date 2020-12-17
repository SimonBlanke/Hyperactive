# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import multiprocessing
from importlib import import_module
from .optimizers import RandomSearchOptimizer


from .run_search import run_search
from .search_info import SearchInfo


def set_n_jobs(n_jobs):
    """Sets the number of jobs to run in parallel"""
    num_cores = multiprocessing.cpu_count()
    if n_jobs == -1 or n_jobs > num_cores:
        return num_cores
    else:
        return n_jobs


def get_class(file_path, class_name):
    module = import_module(file_path, "hyperactive")
    return getattr(module, class_name)


def no_ext_warnings():
    def warn(*args, **kwargs):
        pass

    import warnings

    warnings.warn = warn


"""
def init_optimizer(optimizer, search_space):
    if isinstance(optimizer, dict):
        opt_string = list(optimizer.keys())[0]
        opt_para = optimizer[opt_string]
    else:
        opt_string = optimizer
        opt_para = {}

    module = import_module("gradient_free_optimizers")
    opt_class = getattr(module, optimizer_dict[opt_string])

    search_space_gfo = {}
    for key in search_space.keys():
        dict_value = search_space[key]
        space_dim = np.array(range(len(dict_value)))
        # search_space_pos.append(space_dim)
        search_space_gfo[key] = space_dim

    opt = opt_class(search_space_gfo, **opt_para)

    return opt
"""


class Hyperactive:
    def __init__(
        self,
        verbosity={
            "progress_bar": True,
            "print_results": True,
            "print_times": True,
        },
        distribution="multiprocessing",
    ):
        self.verbosity = verbosity
        self.distribution = distribution

        self.search_processes_infos = {}

    def _add_search_processes(self):
        for nth_job in range(self.n_jobs):
            nth_process = len(self.search_processes_infos)

            self.search_processes_infos[nth_process] = {
                "random_state": self.random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "objective_function": self.objective_function,
                "search_space": self.search_space,
                "optimizer": self.optimizer,
                "n_iter": set_n_jobs(self.n_iter),
                "initialize": self.initialize,
                "memory": self.memory,
            }

    def add_search(
        self,
        objective_function,
        search_space,
        n_iter,
        optimizer=RandomSearchOptimizer(),
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        random_state=None,
        memory=True,
    ):
        self.objective_function = objective_function
        self.search_space = search_space
        self.n_iter = n_iter
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.initialize = initialize
        self.random_state = random_state
        self.memory = memory

        self.optimizer.init(search_space)

        self._add_search_processes()

    def run(self, max_time=None, max_score=None):
        for nth_process in self.search_processes_infos.keys():
            self.search_processes_infos[nth_process]["max_time"] = max_time
            self.search_processes_infos[nth_process]["max_score"] = max_score

        results_list = run_search(
            self.search_processes_infos, self.distribution
        )
