import time
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from hyperactive.optimizers import HillClimbingOptimizer, RandomSearchOptimizer
from hyperactive.experiment import BaseExperiment, add_catch
from hyperactive.search_config import SearchConfig


search_config = SearchConfig(
    x1=list(np.arange(0, 100, 0.1)),
)


def test_max_score_0():
    class Experiment(BaseExperiment):
        def objective_function(self, para):
            score = -para["x1"] * para["x1"]
            return score

    experiment = Experiment()

    max_score = -9999

    hyper = HillClimbingOptimizer(
        epsilon=0.01,
        rand_rest_p=0,
    )
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100000,
        initialize={"warm_start": [{"x1": 99}]},
        max_score=max_score,
    )
    hyper.run()

    print("\n Results head \n", hyper.search_data(experiment).head())
    print("\n Results tail \n", hyper.search_data(experiment).tail())

    print("\nN iter:", len(hyper.search_data(experiment)))

    assert -100 > hyper.best_score(experiment) > max_score


def test_max_score_1():

    class Experiment(BaseExperiment):
        def objective_function(self, para):
            score = -para["x1"] * para["x1"]
            time.sleep(0.01)
            return score

    experiment = Experiment()

    max_score = -9999

    c_time = time.perf_counter()
    hyper = RandomSearchOptimizer()
    hyper.add_search(
        experiment,
        search_config,
        n_iter=100000,
        initialize={"warm_start": [{"x1": 99}]},
        max_score=max_score,
    )
    hyper.run()
    diff_time = time.perf_counter() - c_time

    print("\n Results head \n", hyper.search_data(experiment).head())
    print("\n Results tail \n", hyper.search_data(experiment).tail())

    print("\nN iter:", len(hyper.search_data(experiment)))

    assert diff_time < 1
