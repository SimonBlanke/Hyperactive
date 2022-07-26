# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from ..base_optimizer import BaseOptimizer


class BaseOptimizationStrategy(BaseOptimizer):
    def __init__(self):
        super().__init__()

    def setup_search(
        self,
        objective_function,
        s_space,
        n_iter,
        initialize,
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
        self.pass_through = pass_through
        self.callbacks = callbacks
        self.catch = catch
        self.max_score = max_score
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.memory = memory
        self.memory_warm_start = memory_warm_start
        self.verbosity = verbosity

        self._max_time = None

        if "progress_bar" in self.verbosity:
            self.verbosity = ["progress_bar"]
        else:
            self.verbosity = []

    @property
    def max_time(self):
        return self._max_time

    @max_time.setter
    def max_time(self, value):
        self._max_time = value

        for optimizer_setup in self.optimizer_setup_l:
            optimizer_setup["optimizer"].max_time = value

    def search(self, nth_process):
        for optimizer_setup in self.optimizer_setup_l:
            hyper_opt = optimizer_setup["optimizer"]
            duration = optimizer_setup["duration"]

            n_iter = int(self.n_iter * duration)

            hyper_opt.setup_search(
                objective_function=self.objective_function,
                s_space=self.s_space,
                n_iter=n_iter,
                initialize=self.initialize,
                pass_through=self.pass_through,
                callbacks=self.callbacks,
                catch=self.catch,
                max_score=self.max_score,
                early_stopping=self.early_stopping,
                random_state=self.random_state,
                memory=self.memory,
                memory_warm_start=self.memory_warm_start,
                verbosity=self.verbosity,
            )
            hyper_opt.search(nth_process)

            self._add_result_attributes(
                hyper_opt.best_para,
                hyper_opt.best_score,
                hyper_opt.best_since_iter,
                hyper_opt.eval_times,
                hyper_opt.iter_times,
                hyper_opt.positions,
                hyper_opt.search_data,
                hyper_opt.memory_values_df,
            )


class OptimizationStrategy(BaseOptimizationStrategy):
    def __init__(self):
        super().__init__()

        self.optimizer_setup_l = []

    def add_optimizer(self, optimizer, duration=1):
        optimizer_setup = {
            "optimizer": optimizer,
            "duration": duration,
        }
        self.optimizer_setup_l.append(optimizer_setup)
