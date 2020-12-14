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
        random_state=None,
        verbosity={
            "progress_bar": True,
            "print_results": True,
            "print_times": True,
        },
        ext_warnings=False,
    ):
        self.random_state = random_state
        self.verbosity = verbosity

    def search(
        self,
        objective_function,
        search_space,
        n_iter,
        optimizer=RandomSearchOptimizer(),
        max_time=None,
        max_score=None,
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        memory=True,
    ):
        n_jobs = set_n_jobs(n_jobs)

        optimizer.init(search_space)

        process_info_dict = {}
        processes = []
        for nth_job in range(n_jobs):
            nth_process = len(process_info_dict)
            processes.append(nth_process)

            process_info_dict[nth_process] = {
                "random_state": self.random_state,
                "verbosity": self.verbosity,
                "nth_process": nth_process,
                "objective_function": objective_function,
                "search_space": search_space,
                "optimizer": optimizer,
                "n_iter": n_iter,
                "max_time": max_time,
                "max_score": max_score,
                "initialize": initialize,
                "memory": memory,
            }

        results_list = run_search(process_info_dict)
