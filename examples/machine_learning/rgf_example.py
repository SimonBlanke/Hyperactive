from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from rgf.sklearn import RGFClassifier

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    rgf = RGFClassifier(
        max_leaf=para["max_leaf"],
        reg_depth=para["reg_depth"],
        min_samples_leaf=para["min_samples_leaf"],
        algorithm="RGF_Sib",
        test_interval=100,
        verbose=False,
    )
    scores = cross_val_score(rgf, X, y, cv=3)

    return scores.mean()


search_space = {
    "max_leaf": range(10, 2000, 10),
    "reg_depth": range(1, 21),
    "min_samples_leaf": range(1, 21),
}

hyper = Hyperactive(X, y)
hyper.add_search(model, search_space, n_iter=10)
hyper.run()
