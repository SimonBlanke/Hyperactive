# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .layers.optimizer import OptimizerLayer
from .layers.search_space_pruning import SearchSpacePruningLayer


class BaseOptimizationStrategy(OptimizerLayer, SearchSpacePruningLayer):
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

        self.c_layer_key = None
        self.c_layer_setup = None

        if "progress_bar" in self.verbosity:
            self.verbosity = []
        else:
            self.verbosity = []

        self.distribute_iterations()

    @property
    def max_time(self):
        return self._max_time

    @max_time.setter
    def max_time(self, value):
        self._max_time = value

        for layer_key, layer_setup in self.optimizer_setup_d.items():
            if "optimizer.layer" in layer_key:
                layer_setup["opt-algorithm"].max_time = value

    def distribute_iterations(self):
        n_iter_remain = self.n_iter

        for layer_key, layer_setup in self.optimizer_setup_d.items():
            if "optimizer.layer" in layer_key:
                n_iter_layer = round(
                    self.n_iter * layer_setup["duration"] / self.duration_sum
                )
                if n_iter_layer > n_iter_remain:
                    n_iter_layer = n_iter_remain
                layer_setup["n_iter"] = n_iter_layer

                n_iter_remain -= n_iter_layer

    def search(self, nth_process, p_bar):
        for layer_key, layer_setup in self.optimizer_setup_d.items():
            self.c_layer_key = layer_key
            self.c_layer_setup = layer_setup

            if "optimizer.layer" in layer_key:
                self.run_optimization_layer(nth_process, p_bar)
            elif "search-space-pruning.layer" in layer_key:
                self.run_search_space_pruining()
            else:
                raise ValueError
