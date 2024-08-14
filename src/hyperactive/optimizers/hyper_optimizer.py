# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np

from .objective_function import ObjectiveFunction
from .hyper_gradient_conv import HyperGradientConv
from .optimizer_attributes import OptimizerAttributes
from .constraint import Constraint


class HyperOptimizer(OptimizerAttributes):
    def __init__(self, **opt_params):
        super().__init__()
        self.opt_params = opt_params

    def setup_search(
        self,
        objective_function,
        s_space,
        n_iter,
        initialize,
        constraints,
        pass_through,
        callbacks,
        catch,
        max_score,
        early_stopping,
        random_state,
        memory,
        memory_warm_start,
        verbosity,
    ):
        self.objective_function = objective_function
        self.s_space = s_space
        self.n_iter = n_iter

        self.initialize = initialize
        self.constraints = constraints
        self.pass_through = pass_through
        self.callbacks = callbacks
        self.catch = catch
        self.max_score = max_score
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.verbosity = verbosity

        if "progress_bar" in self.verbosity:
            self.verbosity = ["progress_bar"]
        else:
            self.verbosity = []

    def convert_results2hyper(self):
        self.eval_times = np.array(self.gfo_optimizer.eval_times).sum()
        self.iter_times = np.array(self.gfo_optimizer.iter_times).sum()

        if self.gfo_optimizer.best_para is not None:
            value = self.hg_conv.para2value(self.gfo_optimizer.best_para)
            position = self.hg_conv.position2value(value)
            best_para = self.hg_conv.value2para(position)

            self.best_para = best_para
        else:
            self.best_para = None

        self.best_score = self.gfo_optimizer.best_score
        self.positions = self.gfo_optimizer.search_data

        self.search_data = self.hg_conv.positions2results(self.positions)

        results_dd = self.gfo_optimizer.search_data.drop_duplicates(
            subset=self.s_space.dim_keys, keep="first"
        )
        self.memory_values_df = results_dd[
            self.s_space.dim_keys + ["score"]
        ].reset_index(drop=True)

    def _setup_process(self, nth_process):
        self.nth_process = nth_process

        self.hg_conv = HyperGradientConv(self.s_space)

        initialize = self.hg_conv.conv_initialize(self.initialize)
        search_space_positions = self.s_space.positions

        # conv warm start for smbo from values into positions
        if "warm_start_smbo" in self.opt_params:
            self.opt_params["warm_start_smbo"] = self.hg_conv.conv_memory_warm_start(
                self.opt_params["warm_start_smbo"]
            )

        gfo_constraints = [
            Constraint(constraint, self.s_space) for constraint in self.constraints
        ]

        self.gfo_optimizer = self.optimizer_class(
            search_space=search_space_positions,
            initialize=initialize,
            constraints=gfo_constraints,
            random_state=self.random_state,
            nth_process=nth_process,
            **self.opt_params
        )

        self.conv = self.gfo_optimizer.conv

    def search(self, nth_process, p_bar):
        self._setup_process(nth_process)

        gfo_wrapper_model = ObjectiveFunction(
            objective_function=self.objective_function,
            optimizer=self.gfo_optimizer,
            callbacks=self.callbacks,
            catch=self.catch,
            nth_process=self.nth_process,
        )
        gfo_wrapper_model.pass_through = self.pass_through

        memory_warm_start = self.hg_conv.conv_memory_warm_start(self.memory_warm_start)

        gfo_objective_function = gfo_wrapper_model(self.s_space())

        self.gfo_optimizer.init_search(
            gfo_objective_function,
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
                    + str(nth_process)
                    + "] "
                    + str(self.objective_function.__name__)
                    + " ("
                    + self.optimizer_class.name
                    + ")",
                )

            self.gfo_optimizer.search_step(nth_iter)
            if self.gfo_optimizer.stop.check():
                break

            if p_bar:
                p_bar.set_postfix(
                    best_score=str(gfo_wrapper_model.optimizer.score_best),
                    best_pos=str(gfo_wrapper_model.optimizer.pos_best),
                    best_iter=str(gfo_wrapper_model.optimizer.p_bar._best_since_iter),
                )

                p_bar.update(1)
                p_bar.refresh()

        self.gfo_optimizer.finish_search()

        self.convert_results2hyper()

        self._add_result_attributes(
            self.best_para,
            self.best_score,
            self.gfo_optimizer.p_bar._best_since_iter,
            self.eval_times,
            self.iter_times,
            self.search_data,
            self.gfo_optimizer.random_seed,
        )
