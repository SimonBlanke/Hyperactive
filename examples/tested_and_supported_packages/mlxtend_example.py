from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from hyperactive import Hyperactive


data = load_breast_cancer()
X, y = data.data, data.target


def model(opt):
    dtc = DecisionTreeClassifier(
        min_samples_split=opt["min_samples_split"],
        min_samples_leaf=opt["min_samples_leaf"],
    )
    mlp = MLPClassifier(hidden_layer_sizes=opt["hidden_layer_sizes"])
    svc = SVC(C=opt["C"], degree=opt["degree"], gamma="auto", probability=True)

    eclf = EnsembleVoteClassifier(
        clfs=[dtc, mlp, svc], weights=opt["weights"], voting="soft",
    )

    scores = cross_val_score(eclf, X, y, cv=3)

    return scores.mean()


search_space = {
    "min_samples_split": list(range(2, 15)),
    "min_samples_leaf": list(range(1, 15)),
    "hidden_layer_sizes": list(range(5, 50, 5)),
    "weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
    "C": list(range(1, 1000)),
    "degree": list(range(0, 8)),
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=25)
hyper.run()

