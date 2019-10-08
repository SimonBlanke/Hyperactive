# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive
from hyperactive import MetaLearn

data = load_iris()
X = data.data
y = data.target


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=3)

    return scores.mean(), model


search_config = {
    model: {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}


def test_metalearn():
    ml = MetaLearn(search_config)
    ml.collect(X, y)
    # ml.train()
    # ml.search(X, y)


def test_metalearn1():
    opt = Hyperactive(search_config, meta_learn=True)
    opt.search(X, y)


test_metalearn()
