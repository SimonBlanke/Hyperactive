# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import time
import numpy as np

from .search_space import SearchSpace
from .model import Model
from .init_position import InitSearchPosition

from hypermemory import Hypermemory
from importlib import import_module


def meta_data_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/meta_data/"


optimizer_dict = {
    "HillClimbing": "HillClimbingOptimizer",
    "StochasticHillClimbing": "StochasticHillClimbingOptimizer",
    "TabuSearch": "TabuOptimizer",
    "RandomSearch": "RandomSearchOptimizer",
    "RandomRestartHillClimbing": "RandomRestartHillClimbingOptimizer",
    "RandomAnnealing": "RandomAnnealingOptimizer",
    "SimulatedAnnealing": "SimulatedAnnealingOptimizer",
    "StochasticTunneling": "StochasticTunnelingOptimizer",
    "ParallelTempering": "ParallelTemperingOptimizer",
    "ParticleSwarm": "ParticleSwarmOptimizer",
    "EvolutionStrategy": "EvolutionStrategyOptimizer",
    "Bayesian": "BayesianOptimizer",
    "TPE": "TreeStructuredParzenEstimators",
    "DecisionTree": "DecisionTreeOptimizer",
}


class ShortTermMemory:
    def __init__(self, _space_, _main_args_, _cand_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = {}
        self.memory_dict_new = {}

        self.meta_data_found = False

        self.n_dims = None


class HypermemoryWrapper:
    def __init__(self):
        pass

    def load_memory(self, X, y):
        self.mem = Hypermemory(X, y, self.obj_func, self.search_space,)
        self.eval_pos = self.eval_pos_Mem
        self.memory_dict = self.mem.load()


class SearchProcess:
    def __init__(self, nth_process, pro_arg, verb):
        self.nth_process = nth_process
        self.pro_arg = pro_arg
        self.verb = verb

        kwargs = self.pro_arg.kwargs
        module = import_module("gradient_free_optimizers")

        self.opt_class = getattr(module, optimizer_dict[pro_arg.optimizer])
        self.obj_func = kwargs["objective_function"]
        self.func_para = kwargs["function_parameter"]
        self.search_space = kwargs["search_space"]
        self.n_iter = kwargs["n_iter"]
        self.n_jobs = kwargs["n_jobs"]
        self.memory = kwargs["memory"]
        self.init_para = kwargs["init_para"]
        self.distribution = kwargs["distribution"]

        self.space = SearchSpace(kwargs["search_space"], verb)
        self.model = Model(self.obj_func, self.func_para, verb)
        self.init = InitSearchPosition(self.init_para, self.space, verb)

        self.start_time = time.time()
        self.i = 0

        self.memory_dict = {}
        self.memory_dict_new = {}

        self._score = -np.inf
        self._pos = None

        self.score_best = -np.inf
        self.pos_best = None

        self.score_list = []
        self.pos_list = []

        self.eval_time = []
        self.iter_times = []

        print("self.memory", self.memory)

        self._memory_processor()

    def _memory_processor(self):
        if not self.memory:
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        elif self.memory == "short":
            self.mem = None
            self.eval_pos = self.eval_pos_Mem

        elif self.memory == "long":
            self.mem = Hypermemory(
                self.func_para["features"],
                self.func_para["target"],
                self.obj_func,
                self.search_space,
            )
            self.eval_pos = self.eval_pos_Mem

            self.memory_dict = self.mem.load()

        else:
            print("Warning: Memory not defined")
            self.mem = None
            self.eval_pos = self.eval_pos_noMem

        if self.mem:
            if self.mem.meta_data_found:
                self.pos_best = self.mem.pos_best
                self.score_best = self.mem.score_best

    def _get_warm_start(self):
        return self.space.pos2para(self.pos_best)

    def _process_results(self):
        self.total_time = time.time() - self.start_time
        start_point = self.verb.info.print_start_point(self)

        if self.memory == "long":
            self.mem.dump(self.memory_dict_new)

        return start_point

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self.score_list.append(value)
        self._score = value

    @property
    def pos(self):
        return self._score

    @pos.setter
    def pos(self, value):
        self.pos_list.append(value)
        self._pos = value

    def base_eval(self, pos, nth_iter):
        para = self.space.pos2para(pos)
        para["iteration"] = self.i
        results = self.model.eval(para)

        if results["score"] > self.score_best:
            self.score_best = results["score"]
            self.pos_best = pos

            self.verb.p_bar.best_since_iter = nth_iter

        return results

    def eval_pos_noMem(self, pos, nth_iter):
        results = self.base_eval(pos, nth_iter)
        return results["score"]

    def eval_pos_Mem(self, pos, nth_iter, force_eval=False):
        pos.astype(int)
        pos_tuple = tuple(pos)

        if pos_tuple in self.memory_dict and not force_eval:
            return self.memory_dict[pos_tuple]["score"]
        else:
            results = self.base_eval(pos, nth_iter)
            self.memory_dict[pos_tuple] = results
            self.memory_dict_new[pos_tuple] = results

            return results["score"]

    def _get_score(self, pos_new, nth_iter):
        score_new = self.eval_pos(pos_new, nth_iter)
        self.verb.p_bar.update_p_bar(1, self.score_best)

        if score_new > self.score_best:
            self.score = score_new
            self.pos = pos_new

        return score_new

    def search(self, start_time, max_time, nth_process):
        self._initialize_search(nth_process)

        # loop to initialize N positions
        for nth_init in range(len(self.opt.init_positions)):
            pos_new = self.opt.init_pos(nth_init)
            score_new = self._get_score(pos_new, 0)
            self.opt.evaluate(score_new)

        # loop to do the iterations
        for nth_iter in range(len(self.opt.init_positions), self.n_iter):
            pos_new = self.opt.iterate(nth_iter)
            score_new = self._get_score(pos_new, nth_iter)
            self.opt.evaluate(score_new)

            if self._time_exceeded(start_time, max_time):
                break

        self.verb.p_bar.close_p_bar()

        return self.opt.p_list

    def _time_exceeded(self, start_time, max_time):
        run_time = time.time() - start_time
        return max_time and run_time > max_time

    def _initialize_search(self, nth_process):
        n_positions = self.pro_arg.n_positions
        init_positions = self.init.set_start_pos(n_positions)
        self.opt = self.opt_class(init_positions, self.space.dim, opt_para={})

        self.pro_arg.set_random_seed(nth_process)
        self.verb.p_bar.init_p_bar(nth_process, self.n_iter, self.obj_func)
