from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive


def test_issue_29():
    data = load_diabetes()
    X, y = data.data, data.target

    def model(para):
        dtr = DecisionTreeRegressor(
            min_samples_split=para["min_samples_split"],
            max_depth=para["max_depth"],
        )
        scores = cross_val_score(dtr, X, y, cv=3)

        print(
            "Iteration:",
            para.optimizer.nth_iter,
            " Best score",
            para.optimizer.best_score,
        )

        return scores.mean()

    search_space = {
        "min_samples_split": list(range(2, 12)),
        "max_depth": list(range(2, 12)),
    }

    hyper = Hyperactive()
    hyper.add_search(model, search_space, n_iter=20)
    hyper.run()
