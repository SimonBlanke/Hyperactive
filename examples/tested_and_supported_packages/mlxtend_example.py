from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from hyperactive import Hyperactive


data = load_breast_cancer()
X, y = data.data, data.target


def model(para, X, y):
    dtc = DecisionTreeClassifier(
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    mlp = MLPClassifier(hidden_layer_sizes=para["hidden_layer_sizes"])
    svc = SVC(C=para["C"], degree=para["degree"], gamma="auto", probability=True)

    eclf = EnsembleVoteClassifier(
        clfs=[dtc, mlp, svc], weights=para["weights"], voting="soft"
    )

    scores = cross_val_score(eclf, X, y, cv=3)

    return scores.mean()


search_space = {
    "min_samples_split": range(2, 15),
    "min_samples_leaf": range(1, 15),
    "hidden_layer_sizes": [(x,) for x in range(5, 30)],
    "weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
    "C": range(1, 1000),
    "degree": range(0, 8),
}


hyper = Hyperactive(X, y)
hyper.add_search(model, search_space, n_iter=30)
hyper.run()

