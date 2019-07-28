# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .util import merge_dicts

from sklearn.gaussian_process.kernels import Matern


class Arguments:
    def __init__(self, *args, **kwargs):
        kwargs_opt = {
            # HillClimbingOptimizer
            "eps": 1,
            # StochasticHillClimbingOptimizer
            "r": 1,
            # TabuOptimizer
            "tabu_memory": [3, 6, 9],
            # RandomRestartHillClimbingOptimizer
            "n_restarts": 10,
            # RandomAnnealingOptimizer
            "eps_global": 100,
            "t_rate": 0.98,
            # SimulatedAnnealingOptimizer
            "n_neighbours": 1,
            # StochasticTunnelingOptimizer
            "gamma": 1,
            # ParallelTemperingOptimizer
            "system_temps": [0.1, 0.2, 0.01],
            "n_swaps": 10,
            # ParticleSwarmOptimizer
            "n_part": 10,
            "w": 0.5,
            "c_k": 0.5,
            "c_s": 0.9,
            # EvolutionStrategyOptimizer
            "individuals": 10,
            "mutation_rate": 0.7,
            "crossover_rate": 0.3,
            # BayesianOptimizer
            "kernel": Matern(nu=2.5),
        }

        kwargs_opt = merge_dicts(kwargs_opt, kwargs)

        self._set_specific_args(kwargs_opt)

    def _set_specific_args(self, kwargs_opt):
        self.eps = kwargs_opt["eps"]
        self.r = kwargs_opt["r"]
        self.tabu_memory = kwargs_opt["tabu_memory"]
        self.n_restarts = kwargs_opt["n_restarts"]
        self.eps_global = kwargs_opt["eps_global"]
        self.t_rate = kwargs_opt["t_rate"]
        self.n_neighbours = kwargs_opt["n_neighbours"]
        self.gamma = kwargs_opt["gamma"]
        self.system_temps = kwargs_opt["system_temps"]
        self.n_swaps = kwargs_opt["n_swaps"]
        self.n_part = kwargs_opt["n_part"]
        self.w = kwargs_opt["w"]
        self.c_k = kwargs_opt["c_k"]
        self.c_s = kwargs_opt["c_s"]
        self.individuals = kwargs_opt["individuals"]
        self.mutation_rate = kwargs_opt["mutation_rate"]
        self.crossover_rate = kwargs_opt["crossover_rate"]
        self.kernel = kwargs_opt["kernel"]
