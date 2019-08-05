# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

search_config = {
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}


def test_sklearn():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 3)
    opt.fit(X, y)
    opt.predict(X)
    opt.score(X, y)


def test_sklearn_score():
    from hyperactive import RandomSearchOptimizer

    ml_scores = ["accuracy_score"]

    for score in ml_scores:
        opt = RandomSearchOptimizer(search_config, 3, metric=score)
        assert opt._config_.metric == score
        opt.fit(X, y)
        assert opt._config_.metric == score
        opt.predict(X)
        assert opt._config_.metric == score
        opt.score(X, y)
        assert opt._config_.metric == score


def test_sklearn_loss():
    from hyperactive import RandomSearchOptimizer

    ml_losses = [
        "mean_absolute_error",
        "mean_squared_error",
        "mean_squared_log_error",
        "median_absolute_error",
    ]

    for loss in ml_losses:
        opt = RandomSearchOptimizer(search_config, 3, metric=loss)
        assert opt._config_.metric == loss
        opt.fit(X, y)
        assert opt._config_.metric == loss
        opt.predict(X)
        assert opt._config_.metric == loss
        opt.score(X, y)
        assert opt._config_.metric == loss


"""
def test_sklearn_n_jobs():
    from hyperactive import RandomSearchOptimizer

    n_jobs_list = [1, 2, 3, 4]
    for n_jobs in n_jobs_list:
        opt = RandomSearchOptimizer(search_config, 3, n_jobs=n_jobs)
        assert opt._config_.n_jobs == n_jobs

        assert 2 == 3
            +  where 2 = <hyperactive.config.Config object at 0x7f91a00fedd8>.n_jobs
            +  where <hyperactive.config.Config object at 0x7f91a00fedd8> = <hyperactive.optimizers.random.random_search.RandomSearchOptimizer object at 0x7f91a01445c0>._config_

        opt.fit(X, y)
        assert opt._config_.n_jobs == n_jobs
        opt.predict(X)
        assert opt._config_.n_jobs == n_jobs
        opt.score(X, y)
        assert opt._config_.n_jobs == n_jobs
"""


def test_sklearn_n_iter():
    from hyperactive import RandomSearchOptimizer

    n_iter_list = [0, 1, 3, 10]
    for n_iter in n_iter_list:
        opt = RandomSearchOptimizer(search_config, n_iter)
        assert opt._config_.n_iter == n_iter
        opt.fit(X, y)
        assert opt._config_.n_iter == n_iter
        opt.predict(X)
        assert opt._config_.n_iter == n_iter
        opt.score(X, y)
        assert opt._config_.n_iter == n_iter


def test_sklearn_cv():
    from hyperactive import RandomSearchOptimizer

    cv_list = [0.1, 0.5, 0.9, 2, 4]
    for cv in cv_list:
        opt = RandomSearchOptimizer(search_config, 3, cv=cv)
        assert opt._config_.cv == cv
        opt.fit(X, y)
        assert opt._config_.cv == cv
        opt.predict(X)
        assert opt._config_.cv == cv
        opt.score(X, y)
        assert opt._config_.cv == cv


def test_sklearn_verbosity():
    from hyperactive import RandomSearchOptimizer

    verbosity_list = [0, 1, 2]
    for verbosity in verbosity_list:
        opt = RandomSearchOptimizer(search_config, 3, verbosity=verbosity)
        opt.fit(X, y)
        opt.predict(X)
        opt.score(X, y)


def test_sklearn_random_state():
    from hyperactive import RandomSearchOptimizer

    random_state_list = [None, 0, 1, 2]
    for random_state in random_state_list:
        opt = RandomSearchOptimizer(search_config, 3, random_state=random_state)
        assert opt._config_.random_state == random_state
        opt.fit(X, y)
        assert opt._config_.random_state == random_state
        opt.predict(X)
        assert opt._config_.random_state == random_state
        opt.score(X, y)
        assert opt._config_.random_state == random_state


def test_sklearn_warm_start():
    from hyperactive import RandomSearchOptimizer

    warm_start = {"sklearn.tree.DecisionTreeClassifier": {"max_depth": [1]}}

    warm_start_list = [None, warm_start]
    for warm_start in warm_start_list:
        opt = RandomSearchOptimizer(search_config, 3, warm_start=warm_start)
        assert opt._config_.warm_start == warm_start
        opt.fit(X, y)
        assert opt._config_.warm_start == warm_start
        opt.predict(X)
        assert opt._config_.warm_start == warm_start
        opt.score(X, y)
        assert opt._config_.warm_start == warm_start


def test_sklearn_memory():
    from hyperactive import RandomSearchOptimizer

    memory_list = [False, True]
    for memory in memory_list:
        opt = RandomSearchOptimizer(search_config, 3, memory=memory)
        assert opt._config_.memory == memory
        opt.fit(X, y)
        assert opt._config_.memory == memory
        opt.predict(X)
        assert opt._config_.memory == memory
        opt.score(X, y)
        assert opt._config_.memory == memory


def test_sklearn_scatter_init():
    from hyperactive import RandomSearchOptimizer

    scatter_init_list = [False, 2, 3, 4]
    for scatter_init in scatter_init_list:
        opt = RandomSearchOptimizer(search_config, 3, scatter_init=scatter_init)
        assert opt._config_.scatter_init == scatter_init
        opt.fit(X, y)
        assert opt._config_.scatter_init == scatter_init
        opt.predict(X)
        assert opt._config_.scatter_init == scatter_init
        opt.score(X, y)
        assert opt._config_.scatter_init == scatter_init
