# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from .util import merge_dicts

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import ExtraTreesRegressor as _ExtraTreesRegressor_
from sklearn.ensemble import ExtraTreesRegressor as _RandomForestRegressor_

from sklearn.gaussian_process.kernels import Matern
from numpy.random import normal


def _return_std(X, trees, predictions, min_variance):
    std = np.zeros(len(X))

    for tree in trees:
        var_tree = tree.tree_.impurity[tree.apply(X)]
        var_tree[var_tree < min_variance] = min_variance
        mean_tree = tree.predict(X)
        std += var_tree + mean_tree ** 2

    std /= len(trees)
    std -= predictions ** 2.0
    std[std < 0.0] = 0.0
    std = std ** 0.5
    # print("std", std)
    return std


class RandomForestRegressor(_RandomForestRegressor_):
    def __init__(self, min_variance=0.0, **kwargs):
        self.min_variance = min_variance
        super().__init__(**kwargs)

    def fit(self, X, y):
        super().fit(X, np.ravel(y))

    def predict(self, X, return_std=False):
        mean = super().predict(X)

        if return_std:
            if self.criterion != "mse":
                raise ValueError(
                    "Expected impurity to be 'mse', got %s instead" % self.criterion
                )
            std = _return_std(X, self.estimators_, mean, self.min_variance)
            return mean.reshape(-1, 1), std
        return mean.reshape(-1, 1)


class GPR:
    def __init__(self):
        self.gpr = GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True)

    def fit(self, X, y):
        self.gpr.fit(X, y)

    def predict(self, X, return_std=False):
        return self.gpr.predict(X, return_std=return_std)


class Arguments:
    def __init__(self, *args, **kwargs):
        kwargs_opt = {
            # HillClimbingOptimizer
            "epsilon": 0.05,
            "climb_dist": normal,
            "n_neighbours": 1,
            # StochasticHillClimbingOptimizer
            "p_down": 0.3,
            # TabuOptimizer
            "tabu_memory": 3,
            # RandomRestartHillClimbingOptimizer
            "n_restarts": 10,
            # RandomAnnealingOptimizer
            "epsilon_mod": 10,
            "annealing_rate": 0.99,
            # SimulatedAnnealingOptimizer
            "start_temp": 1,
            "norm_factor": "adaptive",
            # StochasticTunnelingOptimizer
            "gamma": 0.5,
            # ParallelTemperingOptimizer
            "system_temperatures": [0.1, 1, 10, 100],
            "n_swaps": 10,
            # ParticleSwarmOptimizer
            "n_particles": 10,
            "inertia": 0.5,
            "cognitive_weight": 0.5,
            "social_weight": 0.5,
            # EvolutionStrategyOptimizer
            "individuals": 10,
            "mutation_rate": 0.7,
            "crossover_rate": 0.3,
            # BayesianOptimizer
            "warm_start_smbo": False,
            "xi": 0.01,
            "gpr": GPR(),
            # TreeStructuredParzenEstimators
            "start_up_evals": 10,
            "gamme_tpe": 0.3,
            "tree_regressor": RandomForestRegressor(),
        }

        self.kwargs_opt = merge_dicts(kwargs_opt, kwargs)

        self._set_specific_args(self.kwargs_opt)

    def _set_specific_args(self, kwargs_opt):
        self.epsilon = kwargs_opt["epsilon"]
        self.climb_dist = kwargs_opt["climb_dist"]
        self.n_neighbours = kwargs_opt["n_neighbours"]
        self.p_down = kwargs_opt["p_down"]
        self.tabu_memory = kwargs_opt["tabu_memory"]
        self.n_restarts = kwargs_opt["n_restarts"]
        self.epsilon_mod = kwargs_opt["epsilon_mod"]
        self.annealing_rate = kwargs_opt["annealing_rate"]
        self.start_temp = kwargs_opt["start_temp"]
        self.norm_factor = kwargs_opt["norm_factor"]
        self.gamma = kwargs_opt["gamma"]
        self.system_temperatures = kwargs_opt["system_temperatures"]
        self.n_swaps = kwargs_opt["n_swaps"]
        self.n_particles = kwargs_opt["n_particles"]
        self.inertia = kwargs_opt["inertia"]
        self.cognitive_weight = kwargs_opt["cognitive_weight"]
        self.social_weight = kwargs_opt["social_weight"]
        self.individuals = kwargs_opt["individuals"]
        self.mutation_rate = kwargs_opt["mutation_rate"]
        self.crossover_rate = kwargs_opt["crossover_rate"]
        self.warm_start_smbo = kwargs_opt["warm_start_smbo"]
        self.xi = kwargs_opt["xi"]
        self.gpr = kwargs_opt["gpr"]
        self.start_up_evals = kwargs_opt["start_up_evals"]
        self.gamme_tpe = kwargs_opt["gamme_tpe"]
        self.tree_regressor = kwargs_opt["tree_regressor"]
