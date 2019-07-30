# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

n_iter_0 = 0
n_iter_1 = 3
random_state = 0
cv = 2
n_jobs = 2

search_config = {
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}

warm_start = {"sklearn.tree.DecisionTreeClassifier": {"max_depth": [1]}}


def test_sklearn():
    from hyperactive import HillClimbingOptimizer

    opt = HillClimbingOptimizer(
        search_config,
        n_iter_0,
        random_state=random_state,
        verbosity=0,
        cv=cv,
        n_jobs=1,
        warm_start=warm_start,
    )
    opt.fit(X, y)
    opt.predict(X)
    opt.score(X, y)
