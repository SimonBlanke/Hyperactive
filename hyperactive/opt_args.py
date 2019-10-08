# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .util import merge_dicts

from sklearn.gaussian_process.kernels import Matern


class Arguments:
    def __init__(self, *args, **kwargs):
        kwargs_opt = {
            # HillClimbingOptimizer
            "epsilon": 1,
            # StochasticHillClimbingOptimizer
            "p_down": 1,
            # TabuOptimizer
            "tabu_memory": [3, 6, 9],
            # RandomRestartHillClimbingOptimizer
            "n_restarts": 10,
            # RandomAnnealingOptimizer
            "epsilon_start": 66,
            "annealing_rate": 0.95,
            # SimulatedAnnealingOptimizer
            "n_neighbours": 1,
            # StochasticTunnelingOptimizer
            "gamma": 1,
            # ParallelTemperingOptimizer
            "system_temperatures": [0.001, 0.01, 0.1, 1, 10],
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
            "kernel": Matern(nu=2.5),
        }

        kwargs_opt = merge_dicts(kwargs_opt, kwargs)

        self._set_specific_args(kwargs_opt)

    def _set_specific_args(self, kwargs_opt):
        self.epsilon = kwargs_opt["epsilon"]
        self.p_down = kwargs_opt["p_down"]
        self.tabu_memory = kwargs_opt["tabu_memory"]
        self.n_restarts = kwargs_opt["n_restarts"]
        self.epsilon_start = kwargs_opt["epsilon_start"]
        self.annealing_rate = kwargs_opt["annealing_rate"]
        self.n_neighbours = kwargs_opt["n_neighbours"]
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
        self.kernel = kwargs_opt["kernel"]
