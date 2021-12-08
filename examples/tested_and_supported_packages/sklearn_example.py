from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_diabetes
from hyperactive import Hyperactive

data = load_diabetes()
X, y = data.data, data.target


def model(opt):
    gbr = GradientBoostingRegressor(
        n_estimators=opt["n_estimators"],
        max_depth=opt["max_depth"],
        min_samples_split=opt["min_samples_split"],
    )
    scores = cross_val_score(gbr, X, y, cv=3)

    return scores.mean()


search_space = {
    "n_estimators": list(range(10, 150, 5)),
    "max_depth": list(range(2, 12)),
    "min_samples_split": list(range(2, 22)),
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=20)
hyper.run()
