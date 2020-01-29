# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target
memory = False


def model(para, X, y):
    dtc = DecisionTreeClassifier(
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(dtc, X, y, cv=2)

    return scores.mean()


search_config = {
    model: {
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}


def test_results():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config)

    assert len(list(opt.results[model].keys())) == 3


def test_best_scores():
    opt = Hyperactive(X, y, memory=memory)
    opt.search(search_config)

    assert 0 < opt.best_scores[model] < 1
