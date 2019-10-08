import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target

def meta_opt(para, X, y):

    def model(para, X, y):
        model = GradientBoostingClassifier(
            n_estimators=para["n_estimators"],
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
        )
        scores = cross_val_score(model, X, y, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "n_estimators": range(10, 200, 10),
            "max_depth": range(2, 12),
            "min_samples_split": range(2, 12),
        }
    }

    opt = Hyperactive(search_config, optimizer={"StochasticHillClimbing": {"r": para["r"]}})
    opt.search(X, y)

    return score


search_config = {
    meta_opt: {
        "r": np.arange(0, 10, 0.1),
    }
}

opt = Hyperactive(search_config, optimizer="Bayesian", n_iter=3)
opt.search(X, y)
