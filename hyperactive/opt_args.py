# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .util import merge_dicts
from numpy.random import normal

from .optimizers.sequence_model.surrogate_models import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GPR,
)

tree_regressor = {
    "random_forst": RandomForestRegressor(),
    "extra_tree": ExtraTreesRegressor(),
}


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
            "gamma_tpe": 0.3,
            "tree_regressor": "random_forst",
        }

        self.kwargs_opt = merge_dicts(kwargs_opt, kwargs)

    def set_opt_args(self, n_iter):
        self.epsilon = self.kwargs_opt["epsilon"]
        self.climb_dist = self.kwargs_opt["climb_dist"]
        self.n_neighbours = self.kwargs_opt["n_neighbours"]

        self.p_down = self.kwargs_opt["p_down"]

        self.tabu_memory = self.kwargs_opt["tabu_memory"]

        self.n_restarts = self.kwargs_opt["n_restarts"]
        self.n_iter_restart = int(n_iter / self.n_restarts)

        self.epsilon_mod = self.kwargs_opt["epsilon_mod"]
        self.annealing_rate = self.kwargs_opt["annealing_rate"]
        self.start_temp = self.kwargs_opt["start_temp"]
        self.norm_factor = self.kwargs_opt["norm_factor"]
        self.gamma = self.kwargs_opt["gamma"]

        self.system_temperatures = self.kwargs_opt["system_temperatures"]
        self.n_swaps = self.kwargs_opt["n_swaps"]
        self.n_iter_swap = int(n_iter / self.n_swaps)

        self.n_particles = self.kwargs_opt["n_particles"]
        self.inertia = self.kwargs_opt["inertia"]
        self.cognitive_weight = self.kwargs_opt["cognitive_weight"]
        self.social_weight = self.kwargs_opt["social_weight"]

        self.individuals = self.kwargs_opt["individuals"]
        self.mutation_rate = self.kwargs_opt["mutation_rate"]
        self.crossover_rate = self.kwargs_opt["crossover_rate"]

        self.warm_start_smbo = self.kwargs_opt["warm_start_smbo"]
        self.xi = self.kwargs_opt["xi"]
        self.gpr = self.kwargs_opt["gpr"]

        self.start_up_evals = self.kwargs_opt["start_up_evals"]
        self.gamma_tpe = self.kwargs_opt["gamma_tpe"]
        self.tree_regressor = tree_regressor[self.kwargs_opt["tree_regressor"]]
