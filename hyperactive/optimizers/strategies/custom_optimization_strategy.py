# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .optimization_strategy import BaseOptimizationStrategy


class CustomOptimizationStrategy(BaseOptimizationStrategy):
    def __init__(self):
        super().__init__()

        self.optimizer_setup_d = {}
        self.duration_sum = 0

    def add_optimizer(self, optimizer, duration=1, early_stopping=None):
        self.duration_sum += duration
        optimizer_setup = {
            "opt-algorithm": optimizer,
            "duration": duration,
            "early_stopping": early_stopping,
        }
        self.optimizer_setup_d[
            "optimizer.layer." + str(len(self.optimizer_setup_d))
        ] = optimizer_setup

    def prune_search_space(self, margin=0.01):
        self.optimizer_setup_d["search-space-pruning.layer"] = {"margin": margin}
