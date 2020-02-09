# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from hyperactive import Hyperactive
from hyperactive.memory import delete_model, delete_model_dataset

data = load_iris()
X, y = data.data, data.target


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(criterion=para["criterion"])
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


search_config = {model: {"criterion": ["gini"]}}


def test_delete_model():
    delete_model(model)

    opt = Hyperactive(X, y)
    opt.search(search_config)

    delete_model(model)


def test_delete_model_dataset():
    delete_model_dataset(model, X, y)

    opt = Hyperactive(X, y)
    opt.search(search_config)

    delete_model_dataset(model, X, y)
