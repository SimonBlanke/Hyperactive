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
            print("\n")
            hyper_opt = optimizer_setup["optimizer"]
            duration = optimizer_setup["duration"]

            n_iter = round(self.n_iter * duration / self.duration_sum)

            if self.best_para is not None:
                initialize = {}
                if "warm_start" in initialize:
                    initialize["warm_start"].append(self.best_para)
                else:
                    initialize["warm_start"] = [self.best_para]
            else:
                initialize = dict(self.initialize)

            if self.search_data is not None:
                memory_warm_start = self.search_data
            else:
                memory_warm_start = self.memory_warm_start

            print("\n hyper_opt \n", hyper_opt.__dict__, "\n")
            print("\n optimizer_class \n", hyper_opt.optimizer_class.__dict__, "\n")
            if (
                hyper_opt.optimizer_class.optimizer_type == "sequential"
                and self.search_data is not None
            ):
                hyper_opt.optimizer_class.opt_params[
                    "warm_start_smbo"
                ] = self.search_data

            hyper_opt.setup_search(
                objective_function=self.objective_function,
                s_space=self.s_space,
                n_iter=n_iter,
                initialize=initialize,
                pass_through=self.pass_through,
                callbacks=self.callbacks,
                catch=self.catch,
                max_score=self.max_score,
                early_stopping=self.early_stopping,
                random_state=self.random_state,
                memory=self.memory,
                memory_warm_start=memory_warm_start,
                verbosity=self.verbosity,
            )

            hyper_opt.search(nth_process)

            self._add_result_attributes(
                hyper_opt.best_para,
                hyper_opt.best_score,
                hyper_opt.best_since_iter,
                hyper_opt.eval_times,
                hyper_opt.iter_times,
                hyper_opt.search_data,
                hyper_opt.gfo_optimizer.random_seed,
            )

            print("\n self.best_para \n", self.best_para)
            print(" self.best_score \n", self.best_score)
            print(" self.search_data \n", self.search_data)


class OptimizationStrategy(BaseOptimizationStrategy):
    def __init__(self):
        super().__init__()

        self.optimizer_setup_l = []
        self.duration_sum = 0

    def add_optimizer(self, optimizer, duration=1):
        self.duration_sum += duration
        optimizer_setup = {
            "optimizer": optimizer,
            "duration": duration,
        }
        self.optimizer_setup_l.append(optimizer_setup)
