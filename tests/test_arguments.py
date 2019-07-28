# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

search_config = {"sklearn.tree.DecisionTreeClassifier": {"max_depth": range(1, 21)}}


def test_HillClimbingOptimizer_args():
    from hyperactive import HillClimbingOptimizer

    opt = HillClimbingOptimizer(search_config, 3, eps=2)
    opt.fit(X, y)
