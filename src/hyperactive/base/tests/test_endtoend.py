"""Integration tests for end-to-end usage of optimizers with experiments.

API unit tests are in TestAllOptimizers and TestAllExperiments.
"""
# copyright: hyperactive developers, MIT License (see LICENSE file)


def test_endtoend_hillclimbing()
    """Test end-to-end usage of HillClimbing optimizer with an experiment."""
    # 1. define the experiment
    from hyperactive.experiment.integrations import SklearnCvExperiment
    from sklearn.datasets import load_iris
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold

    X, y = load_iris(return_X_y=True)

    sklearn_exp = SklearnCvExperiment(
        estimator=SVC(),
        scoring=accuracy_score,
        cv=KFold(n_splits=3, shuffle=True),
        X=X,
        y=y,
    )

    # 2. set up the HillClimbing optimizer
    import numpy as np
    from hyperactive.opt.gfo import HillClimbing

    hillclimbing_config = {
        "search_space": {
        "C": np.array([0.01, 0.1, 1, 10]),
        "gamma": np.array([0.0001, 0.01, 0.1, 1, 10]),
        },
        "n_iter": 100,
    }
    hill_climbing = HillClimbing(**hillclimbing_config, experiment=sklearn_exp)

    # 3. run the HillClimbing optimizer
    hill_climbing.run()