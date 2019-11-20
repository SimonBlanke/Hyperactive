from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

import ray

data = load_breast_cancer()
X, y = data.data, data.target


def gbc_(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
    )
    scores = cross_val_score(model, X, y)

    return scores.mean()


search_config = {
    gbc_: {
        "n_estimators": range(1, 20, 1),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
    }
}


ray.init(num_cpus=4)

opt = Hyperactive(X, y)
opt.search(search_config, n_jobs=4)
