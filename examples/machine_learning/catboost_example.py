from sklearn.model_selection import cross_val_score
from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    cbc = CatBoostClassifier(
        iterations=10, depth=para["depth"], learning_rate=para["learning_rate"]
    )
    scores = cross_val_score(cbc, X, y, cv=3)

    return scores.mean()


search_space = {
    "depth": range(2, 15),
    "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
}


hyper = Hyperactive(X, y)
hyper.add_search(model, search_space, n_iter=10)
hyper.run()
