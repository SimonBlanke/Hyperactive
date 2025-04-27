# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ._optimizer_attributes import OptimizerAttributes
from ._constraint import Constraint

from .gradient_free_optimizers.adapter._hyper_gradient_conv import (
    HyperGradientConv,
)
from .gradient_free_optimizers.adapter import Adapter
from .gradient_free_optimizers._runner import Runner


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

    def _setup_process(self):
        self.hg_conv = HyperGradientConv(self.s_space)

        """
        self.gfo_optimizer = self.optimizer_class(
            search_space=search_space_positions,
            initialize=initialize,
            constraints=gfo_constraints,
            random_state=self.random_state,
            nth_process=self.nth_process,
            **self.opt_params,
        )
        """
        # self.conv = self.gfo_optimizer.conv

    def _search(self, p_bar):
        self._setup_process()

        memory_warm_start = self.hg_conv.conv_memory_warm_start(self.memory_warm_start)

        self.experiment.backend_adapter(self.adapter.objective_function, self.s_space)

        self.runner.run_search(self.search_info, self.nth_process, self.max_time, p_bar)
        """
        self.gfo_optimizer.init_search(
            self.experiment.gfo_objective_function,
            self.n_iter,
            self.max_time,
            self.max_score,
            self.early_stopping,
            self.memory,
            memory_warm_start,
            False,
        )
        for nth_iter in range(self.n_iter):
            if p_bar:
                p_bar.set_description(
                    "["
                    + str(self.nth_process)
                    + "] "
                    + str(self.experiment.__class__.__name__)
                    + " ("
                    + self.optimizer_class.name
                    + ")",
                )

            self.gfo_optimizer.search_step(nth_iter)
            if self.gfo_optimizer.stop.check():
                break

            if p_bar:
                p_bar.set_postfix(
                    best_score=str(self.gfo_optimizer.score_best),
                    best_pos=str(self.gfo_optimizer.pos_best),
                    best_iter=str(self.gfo_optimizer.p_bar._best_since_iter),
                )

                p_bar.update(1)
                p_bar.refresh()

        self.gfo_optimizer.finish_search()
        """

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
