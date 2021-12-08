from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive


def test_issue_29():
    data = load_diabetes()
    X, y = data.data, data.target

    def model(para):
        gbr = GradientBoostingRegressor(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
        )
        scores = cross_val_score(gbr, X, y, cv=3)

        print(
            "Iteration:",
            para.optimizer.nth_iter,
            " Best score",
            para.optimizer.best_score,
        )

        return scores.mean()

    search_space = {
        "n_estimators": list(range(10, 150, 5)),
        "max_depth": list(range(2, 12)),
        "min_samples_split": list(range(2, 22)),
    }

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=20)
    hyper.run()
