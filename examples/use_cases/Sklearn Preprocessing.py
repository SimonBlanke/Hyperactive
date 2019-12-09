import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def pca(X):
    X = PCA(n_components=10).fit_transform(X)

    return X


def none(X):
    return X


def model(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"],
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )

    X_pca = para["decomposition"](X)
    X = np.hstack((X, X_pca))

    X = SelectKBest(f_classif, k=para["k"]).fit_transform(X, y)
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {
    model: {
        "decomposition": [pca, none],
        "k": range(2, 30),
        "n_estimators": range(10, 200, 10),
        "max_depth": range(2, 12),
        "min_samples_split": range(2, 12),
        "min_samples_leaf": range(1, 11),
    }
}


opt = Hyperactive(X, y)
opt.search(search_config, n_iter=100)
