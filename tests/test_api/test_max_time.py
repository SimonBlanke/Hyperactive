import time
import numpy as np
from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.experiment import BaseExperiment
from hyperactive.search_config import SearchConfig


class Experiment(BaseExperiment):
    def objective_function(self, para):
        score = -para["x1"] * para["x1"]
        return score


experiment = Experiment()

search_config = SearchConfig(
    x1=list(np.arange(0, 100000, 1)),
)


def test_max_time_0():
    c_time1 = time.perf_counter()
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=1000000)
    hyper.run(max_time=0.1)
    diff_time1 = time.perf_counter() - c_time1

    assert diff_time1 < 1


def test_max_time_1():
    c_time1 = time.perf_counter()
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=1000000)
    hyper.run(max_time=1)
    diff_time1 = time.perf_counter() - c_time1

    assert 0.3 < diff_time1 < 2
