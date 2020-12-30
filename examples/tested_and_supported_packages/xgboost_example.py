from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(opt):
    xgb = XGBClassifier(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        learning_rate=opt["learning_rate"],
    )
    scores = cross_val_score(xgb, X, y, cv=3)

    return scores.mean()


search_space = {
    "n_estimators": list(range(10, 200, 10)),
    "max_depth": list(range(2, 12)),
    "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=30)
hyper.run()
