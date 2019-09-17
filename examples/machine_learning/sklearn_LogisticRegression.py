from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from hyperactive import Hyperactive

iris_data = load_iris()
X = iris_data.data
y = iris_data.target


def model(para, X_train, y_train):
    model = LogisticRegression(
        C=para["C"],
        dual=para["dual"],
        penalty=para["penalty"],
        solver=para["solver"],
        multi_class=para["multi_class"],
        max_iter=para["max_iter"],
    )
    scores = cross_val_score(model, X_train, y_train, cv=3)

    return scores.mean(), model


# this defines the model and hyperparameter search space
search_config = {
    model: {
        "penalty": ["l1", "l2"],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "dual": [False],
        "solver": ["liblinear"],
        "multi_class": ["auto", "ovr"],
        "max_iter": range(300, 1000, 10),
    }
}


opt = Hyperactive(search_config, n_iter=100, n_jobs=2)

# search best hyperparameter for given data
opt.fit(X, y)
