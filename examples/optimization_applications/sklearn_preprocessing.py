import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def pca(X, para):
    X = PCA(n_components=para["n_components"]).fit_transform(X)

    return X


def none(X, para):
    return X


def model(para, X, y):
    model = GradientBoostingClassifier(
        n_estimators=para["n_estimators"], max_depth=para["max_depth"],
    )

    X_pca = para["decomposition"](X, para)
    X = np.hstack((X, X_pca))

    X = SelectKBest(f_classif, k=para["k"]).fit_transform(X, y)
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_space = {
    "decomposition": [pca, none],
    "k": range(2, 30),
    "n_components": range(1, 11),
    "n_estimators": range(10, 100, 3),
    "max_depth": range(2, 12),
}


hyper = Hyperactive(X, y)
hyper.add_search(model, search_space, n_iter=20)
hyper.run()
