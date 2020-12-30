from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from rgf.sklearn import RGFClassifier

from hyperactive import Hyperactive

data = load_breast_cancer()
X, y = data.data, data.target


def model(opt):
    rgf = RGFClassifier(
        max_leaf=opt["max_leaf"],
        reg_depth=opt["reg_depth"],
        min_samples_leaf=opt["min_samples_leaf"],
        algorithm="RGF_Sib",
        test_interval=100,
        verbose=False,
    )
    scores = cross_val_score(rgf, X, y, cv=3)

    return scores.mean()


search_space = {
    "max_leaf": list(range(10, 2000, 10)),
    "reg_depth": list(range(1, 21)),
    "min_samples_leaf": list(range(1, 21)),
}

hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=10)
hyper.run()
