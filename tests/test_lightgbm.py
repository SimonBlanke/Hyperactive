# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

search_config = {
    "lightgbm.LGBMClassifier": {
        "boosting_type": ["gbdt"],
        "num_leaves": range(2, 10),
        "learning_rate": [0.05],
        "feature_fraction": [0.9],
        "bagging_fraction": [0.8],
        "bagging_freq": [5],
    }
}


def test_lightgbm():
    from hyperactive import HillClimbingOptimizer

    opt = HillClimbingOptimizer(search_config, 1)
    opt.fit(X, y)
    opt.predict(X)
    opt.score(X, y)


def test_lightgbm_score():
    from hyperactive import HillClimbingOptimizer

    ml_scores = ["accuracy_score"]

    for score in ml_scores:
        opt = HillClimbingOptimizer(search_config, 1, metric=score)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_lightgbm_loss():
    from hyperactive import HillClimbingOptimizer

    ml_losses = [
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
    ]

    for loss in ml_losses:
        opt = HillClimbingOptimizer(search_config, 1, metric=loss)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


"""
def test_lightgbm_n_jobs():
    from hyperactive import HillClimbingOptimizer

    n_jobs_list = [1, 2, 3, 4]
    for n_jobs in n_jobs_list:
        opt = HillClimbingOptimizer(search_config, 1, n_jobs=n_jobs)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)
"""


def test_lightgbm_n_iter():
    from hyperactive import HillClimbingOptimizer

    n_iter_list = [0, 1, 3, 10]
    for n_iter in n_iter_list:
        opt = HillClimbingOptimizer(search_config, n_iter)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_lightgbm_cv():
    from hyperactive import HillClimbingOptimizer

    cv_list = [0.1, 0.5, 0.9, 2, 4]
    for cv in cv_list:
        opt = HillClimbingOptimizer(search_config, 1, cv=cv)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_lightgbm_verbosity():
    from hyperactive import HillClimbingOptimizer

    verbosity_list = [0, 1, 2]
    for verbosity in verbosity_list:
        opt = HillClimbingOptimizer(search_config, 1, verbosity=verbosity)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_lightgbm_random_state():
    from hyperactive import HillClimbingOptimizer

    random_state_list = [None, 0, 1, 2]
    for random_state in random_state_list:
        opt = HillClimbingOptimizer(search_config, 1, random_state=random_state)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_lightgbm_warm_start():
    from hyperactive import HillClimbingOptimizer

    warm_start = {
        "lightgbm.LGBMClassifier": {
            "boosting_type": ["gbdt"],
            "num_leaves": [5],
            "learning_rate": [0.05],
        }
    }

    warm_start_list = [None, warm_start]
    for warm_start in warm_start_list:
        opt = HillClimbingOptimizer(search_config, 1, warm_start=warm_start)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_lightgbm_memory():
    from hyperactive import HillClimbingOptimizer

    memory_list = [False, True]
    for memory in memory_list:
        opt = HillClimbingOptimizer(search_config, 1, memory=memory)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_lightgbm_scatter_init():
    from hyperactive import HillClimbingOptimizer

    scatter_init_list = [False, 2, 3, 4]
    for scatter_init in scatter_init_list:
        opt = HillClimbingOptimizer(search_config, 1, scatter_init=scatter_init)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)
