import numpy as np
from tqdm import tqdm

from hyperactive.optimizers import HillClimbingOptimizer
from hyperactive.experiment import BaseExperiment
from hyperactive.search_config import SearchConfig


class Experiment(BaseExperiment):
    def objective_function(self, opt):
        score = -opt["x1"] * opt["x1"]
        return score


experiment = Experiment()

search_config = SearchConfig(
    x1=list(np.arange(-100, 101, 1)),
)

n_iter = 15


def test_n_jobs_0():
    n_jobs = 2

    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)
    hyper.run()

    assert len(hyper.search_data(experiment)) == n_iter * n_jobs


def test_n_jobs_1():
    n_jobs = 4

    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=15, n_jobs=n_jobs)
    hyper.run()

    assert len(hyper.search_data(experiment)) == n_iter * n_jobs


def test_n_jobs_2():
    n_jobs = 8

    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)
    hyper.run()

    assert len(hyper.search_data(experiment)) == n_iter * n_jobs


def test_n_jobs_5():
    n_jobs = 2

    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)

    hyper.run()

    assert len(hyper.search_data(experiment)) == n_iter * n_jobs * 2


def test_n_jobs_6():
    n_jobs = 2

    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)
    hyper.add_search(experiment, search_config, n_iter=n_iter, n_jobs=n_jobs)

    hyper.run()

    assert len(hyper.search_data(experiment)) == n_iter * n_jobs * 4


def test_n_jobs_7():
    n_jobs = -1

    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=15, n_jobs=n_jobs)
    hyper.run()


def test_multiprocessing_0():
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=15, n_jobs=2)
    hyper.run(distribution="multiprocessing")


def test_multiprocessing_1():
    hyper = HillClimbingOptimizer()
    hyper.add_search(experiment, search_config, n_iter=15, n_jobs=2)
    hyper.run(
        distribution={
            "multiprocessing": {
                "initializer": tqdm.set_lock,
                "initargs": (tqdm.get_lock(),),
            }
        },
    )
