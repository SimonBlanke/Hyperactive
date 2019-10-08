import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def meta_opt(para, X, y):
    def model(para, X, y):
        model = DecisionTreeClassifier(
            max_depth=para["max_depth"],
            min_samples_split=para["min_samples_split"],
            min_samples_leaf=para["min_samples_leaf"],
        )
        scores = cross_val_score(model, X, y, cv=3)

        return scores.mean()

    search_config = {
        model: {
            "max_depth": range(2, 50),
            "min_samples_split": range(2, 50),
            "min_samples_leaf": range(1, 50),
        }
    }

    opt = Hyperactive(
        search_config,
        optimizer={
            "ParticleSwarm": {"w": para["w"], "c_k": para["c_k"], "c_s": para["c_s"]}
        },
        verbosity=None,
    )
    opt.search(X, y)

    return opt.score_best


search_config = {
    meta_opt: {
        "w": np.arange(0, 1, 0.01),
        "c_k": np.arange(0, 1, 0.01),
        "c_s": np.arange(0, 1, 0.01),
    }
}

opt = Hyperactive(search_config, optimizer="Bayesian", n_iter=50)
opt.search(X, y)
