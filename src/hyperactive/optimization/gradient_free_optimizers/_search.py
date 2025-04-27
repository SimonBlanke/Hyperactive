# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ._optimizer_attributes import OptimizerAttributes
from ._constraint import Constraint

from .adapter._hyper_gradient_conv import (
    HyperGradientConv,
)
from .adapter import Adapter
from ._runner import Runner


class Search(OptimizerAttributes):
    max_time: float
    nth_process: int

    def __init__(self, optimizer_class, opt_params):
        super().__init__()
        self.optimizer_class = optimizer_class
        self.opt_params = opt_params

        self.runner = Runner(optimizer_class, opt_params)

    def setup(self, search_info):
        self.search_info = search_info

        self.adapter = Adapter(search_info)

        self.experiment = search_info.experiment
        self.s_space = search_info.s_space
        self.n_iter = search_info.n_iter

        self.initialize = search_info.initialize
        self.constraints = search_info.constraints
        self.max_score = search_info.max_score
        self.early_stopping = search_info.early_stopping
        self.random_state = search_info.random_state
        self.memory = search_info.memory
        self.memory_warm_start = search_info.memory_warm_start

    def pass_args(self, max_time, nth_process, verbosity):
        self.max_time = max_time
        self.nth_process = nth_process

        if "progress_bar" in verbosity:
            self.verbosity = ["progress_bar"]
        else:
            self.verbosity = []

    def _search(self, p_bar):
        self.experiment.backend_adapter(self.adapter.objective_function, self.s_space)

        self.runner.run_search(self.search_info, self.nth_process, self.max_time, p_bar)

        self.runner.convert_results2hyper()

        self._add_result_attributes(
            self.runner.best_para,
            self.runner.best_score,
            self.runner.gfo_optimizer.p_bar._best_since_iter,
            self.runner.eval_times,
            self.runner.iter_times,
            self.runner.search_data,
            self.runner.gfo_optimizer.random_seed,
        )
