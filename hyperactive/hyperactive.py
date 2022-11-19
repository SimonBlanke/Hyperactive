# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import multiprocessing as mp

from .optimizers import RandomSearchOptimizer
from .run_search import run_search

from .results import Results
from .print_results import PrintResults
from .search_space import SearchSpace


class Hyperactive:
    def __init__(
        self,
        verbosity=["progress_bar", "print_results", "print_times"],
        distribution="multiprocessing",
        n_processes="auto",
    ):
        super().__init__()
        if verbosity is False:
            verbosity = []

        self.verbosity = verbosity
        self.distribution = distribution
        self.n_processes = n_processes

        self.opt_pros = {}

    def _create_shared_memory(self, new_opt):
        if new_opt.memory == "share":
            if len(self.opt_pros) == 0:

                manager = mp.Manager()
                new_opt.memory = manager.dict()

            for opt in self.opt_pros.values():
                same_obj_func = (
                    opt.objective_function.__name__
                    == new_opt.objective_function.__name__
                )
                same_ss_length = len(opt.s_space()) == len(new_opt.s_space())

                if same_obj_func and same_ss_length:
                    new_opt.memory = opt.memory  # get same manager.dict
                else:
                    manager = mp.Manager()  # get new manager.dict
                    new_opt.memory = manager.dict()

    @staticmethod
    def _default_opt(optimizer):
        if isinstance(optimizer, str):
            if optimizer == "default":
                optimizer = RandomSearchOptimizer()
        return optimizer

    @staticmethod
    def _default_search_id(search_id, objective_function):
        if not search_id:
            search_id = objective_function.__name__
        return search_id

    @staticmethod
    def check_list(search_space):
        for key in search_space.keys():
            search_dim = search_space[key]

            error_msg = (
                "Value in '{}' of search space dictionary must be of type list".format(
                    key
                )
            )
            if not isinstance(search_dim, list):
                print("Warning", error_msg)
                # raise ValueError(error_msg)

    def add_search(
        self,
        objective_function,
        search_space,
        n_iter,
        search_id=None,
        optimizer="default",
        n_jobs=1,
        initialize={"grid": 4, "random": 2, "vertices": 4},
        pass_through={},
        callbacks={},
        catch={},
        max_score=None,
        early_stopping=None,
        random_state=None,
        memory="share",
        memory_warm_start=None,
    ):
        self.check_list(search_space)

        optimizer = self._default_opt(optimizer)
        search_id = self._default_search_id(search_id, objective_function)
        s_space = SearchSpace(search_space)

        optimizer.setup_search(
            objective_function=objective_function,
            s_space=s_space,
            n_iter=n_iter,
            initialize=initialize,
            pass_through=pass_through,
            callbacks=callbacks,
            catch=catch,
            max_score=max_score,
            early_stopping=early_stopping,
            random_state=random_state,
            memory=memory,
            memory_warm_start=memory_warm_start,
            verbosity=self.verbosity,
        )

        if memory == "share":
            self._create_shared_memory(optimizer)

        if n_jobs == -1:
            n_jobs = mp.cpu_count()

        for _ in range(n_jobs):
            nth_process = len(self.opt_pros)
            self.opt_pros[nth_process] = optimizer

    def _print_info(self):
        print_res = PrintResults(self.opt_pros, self.verbosity)

        for results in self.results_list:
            nth_process = results["nth_process"]
            print_res.print_process(results, nth_process)

    def run(self, max_time=None):
        for opt in self.opt_pros.values():
            opt.max_time = max_time

        self.results_list = run_search(
            self.opt_pros, self.distribution, self.n_processes
        )

        self.results_ = Results(self.results_list, self.opt_pros)

        self._print_info()

    def best_para(self, id_):
        return self.results_.best_para(id_)

    def best_score(self, id_):
        return self.results_.best_score(id_)

    def search_data(self, id_, times=False):
        search_data_ = self.results_.search_data(id_)

        if times == False:
            search_data_.drop(
                labels=["eval_times", "iter_times"],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return search_data_
