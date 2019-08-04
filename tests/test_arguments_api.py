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
search_config_1 = {
    "sklearn.tree.DecisionTreeClassifier": {"max_depth": range(1, 21)},
    "sklearn.neighbors.KNeighborsClassifier": {
        "n_neighbors": range(1, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
}
warm_start = {"sklearn.tree.DecisionTreeClassifier": {"max_depth": [1]}}

warm_start_1 = {
    "sklearn.tree.DecisionTreeClassifier.0": {"max_depth": [1]},
    "sklearn.tree.DecisionTreeClassifier.1": {"max_depth": [2]},
}


def test_multiple_models_one_job():

    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config_1, 1, n_jobs=1)
    opt.fit(X, y)


def test_n_jobs_1():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 1, n_jobs=1)
    opt.fit(X, y)


def test_n_jobs_2():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 1, n_jobs=2)
    opt.fit(X, y)


def test_n_jobs_4():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 1, n_jobs=4)
    opt.fit(X, y)


def test_positional_args():
    from hyperactive import RandomSearchOptimizer

    opt0 = RandomSearchOptimizer(search_config, 1, random_state=False)
    opt0.fit(X, y)

    opt1 = RandomSearchOptimizer(search_config, n_iter=1, random_state=1)
    opt1.fit(X, y)

    opt2 = RandomSearchOptimizer(search_config=search_config, n_iter=1, random_state=1)
    opt2.fit(X, y)


def test_random_state():
    from hyperactive import RandomSearchOptimizer

    opt0 = RandomSearchOptimizer(search_config, 1, random_state=False)
    opt0.fit(X, y)

    opt1 = RandomSearchOptimizer(search_config, 1, random_state=0)
    opt1.fit(X, y)

    opt2 = RandomSearchOptimizer(search_config, 1, random_state=1)
    opt2.fit(X, y)


def test_memory():
    from hyperactive import RandomSearchOptimizer

    opt0 = RandomSearchOptimizer(search_config, 1, memory=True)
    opt0.fit(X, y)

    opt1 = RandomSearchOptimizer(search_config, 1, memory=False)
    opt1.fit(X, y)


def test_verbosity():
    from hyperactive import RandomSearchOptimizer

    opt0 = RandomSearchOptimizer(search_config, 1, verbosity=0)
    opt0.fit(X, y)

    opt1 = RandomSearchOptimizer(search_config, 1, verbosity=1)
    opt1.fit(X, y)


def test_metrics():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(
        search_config, 1, metric="mean_absolute_error", n_jobs=1
    )
    opt.fit(X, y)


def test_scatter_init():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 1, n_jobs=1, scatter_init=10)
    opt.fit(X, y)


def test_scatter_init_and_warm_start():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(
        search_config, 1, warm_start=warm_start, scatter_init=10
    )
    opt.fit(X, y)

    opt = RandomSearchOptimizer(
        search_config, 2, warm_start=warm_start, scatter_init=10
    )
    opt.fit(X, y)


def test_warm_starts():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 1, n_jobs=1, warm_start=warm_start_1)
    opt.fit(X, y)


def test_warm_start_multiple_jobs():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 1, n_jobs=4, warm_start=warm_start)
    opt.fit(X, y)


def test_warm_start():
    from hyperactive import RandomSearchOptimizer

    opt = RandomSearchOptimizer(search_config, 1, n_jobs=1, warm_start=warm_start)
    opt.fit(X, y)
