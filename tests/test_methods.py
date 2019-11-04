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

n_iter = 1


def model(para, X_train, y_train):
    model = DecisionTreeClassifier(
        criterion=para["criterion"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=2)

    return scores.mean()


search_config = {
    model: {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 11),
        "min_samples_leaf": range(1, 11),
    }
}


def test_get_total_time():
    opt = Hyperactive(search_config, n_iter=n_iter, optimizer="HillClimbing")
    opt.search(X, y)
    opt.get_total_time()


def test_get_eval_time():
    opt = Hyperactive(search_config, n_iter=n_iter, optimizer="HillClimbing")
    opt.search(X, y)
    opt.get_eval_time()


def test_save_report():
    opt = Hyperactive(search_config, n_iter=n_iter, optimizer="HillClimbing")
    opt.search(X, y)
    opt.save_report()
