from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from hyperactive import Hyperactive

data = load_boston()
X, y = data.data, data.target


def model(para, X, y):
    gbr = GradientBoostingRegressor(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=3)

    return scores.mean()


search_space = {
    "n_estimators": range(10, 150, 5),
    "max_depth": range(2, 12),
    "min_samples_split": range(2, 22),
}


hyper = Hyperactive(X, y)
hyper.add_search(model, search_space, n_iter=20)
hyper.run()
