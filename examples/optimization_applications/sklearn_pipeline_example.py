from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def pipeline1(filter_, gbc):
    return Pipeline([("filter_", filter_), ("gbc", gbc)])


def pipeline2(filter_, gbc):
    return gbc


def model(opt):
    gbc = GradientBoostingClassifier(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
    )
    filter_ = SelectKBest(f_classif, k=opt["k"])
    model_ = opt["pipeline"](filter_, gbc)

    scores = cross_val_score(model_, X, y, cv=3)

    return scores.mean()


search_space = {
    "k": list(range(2, 30)),
    "n_estimators": list(range(10, 200, 10)),
    "max_depth": list(range(2, 12)),
    "min_samples_split": list(range(2, 12)),
    "min_samples_leaf": list(range(1, 11)),
    "pipeline": [pipeline1, pipeline2],
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=30)
hyper.run()
