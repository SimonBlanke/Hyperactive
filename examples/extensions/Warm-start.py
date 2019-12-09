from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(gbc, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}

init_config = {
    model: {"n_estimators": [190], "max_depth": [2], "min_samples_split": [5]}
}


opt = Hyperactive(search_config, init_config=init_config)
opt.search(X, y)
