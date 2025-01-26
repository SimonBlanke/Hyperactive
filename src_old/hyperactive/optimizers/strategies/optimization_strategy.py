# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .optimizer_attributes import OptimizerAttributes


class BaseOptimizationStrategy(OptimizerAttributes):
    def __init__(self):
        super().__init__()

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

        self._max_time = None

        if "progress_bar" in self.verbosity:
            self.verbosity = []
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

    def search(self, nth_process, p_bar):
        for optimizer_setup in self.optimizer_setup_l:
            hyper_opt = optimizer_setup["optimizer"]
            duration = optimizer_setup["duration"]
            opt_strat_early_stopping = optimizer_setup["early_stopping"]

            if opt_strat_early_stopping:
                early_stopping = opt_strat_early_stopping
            else:
                early_stopping = self.early_stopping

            n_iter = round(self.n_iter * duration / self.duration_sum)

            # initialize
            if self.best_para is not None:
                initialize = {}
                if "warm_start" in initialize:
                    initialize["warm_start"].append(self.best_para)
                else:
                    initialize["warm_start"] = [self.best_para]
            else:
                initialize = dict(self.initialize)

            # memory_warm_start
            if self.search_data is not None:
                memory_warm_start = self.search_data
            else:
                memory_warm_start = self.memory_warm_start

            # warm_start_smbo
            if (
                hyper_opt.optimizer_class.optimizer_type == "sequential"
                and self.search_data is not None
            ):
                hyper_opt.opt_params["warm_start_smbo"] = self.search_data

            hyper_opt.setup_search(
                objective_function=self.objective_function,
                s_space=self.s_space,
                n_iter=n_iter,
                initialize=initialize,
                constraints=self.constraints,
                pass_through=self.pass_through,
                callbacks=self.callbacks,
                catch=self.catch,
                max_score=self.max_score,
                early_stopping=early_stopping,
                random_state=self.random_state,
                memory=self.memory,
                memory_warm_start=memory_warm_start,
                verbosity=self.verbosity,
            )

            hyper_opt.search(nth_process, p_bar)

            self._add_result_attributes(
                hyper_opt.best_para,
                hyper_opt.best_score,
                hyper_opt.best_since_iter,
                hyper_opt.eval_times,
                hyper_opt.iter_times,
                hyper_opt.search_data,
                hyper_opt.gfo_optimizer.random_seed,
            )
