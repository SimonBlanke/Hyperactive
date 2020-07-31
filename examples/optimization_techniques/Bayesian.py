from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive

import GPy

data = load_iris()
X, y = data.data, data.target


def model(para, X, y):
    knr = KNeighborsClassifier(n_neighbors=para["n_neighbors"])
    scores = cross_val_score(knr, X, y, cv=5)
    score = scores.mean()

    return score


search_space = {
    "n_neighbors": list(range(1, 100)),
}

optimizer = "Bayesian"

hyper = Hyperactive(X, y)
hyper.add_search(model, search_space, optimizer=optimizer, n_iter=100)
hyper.run()


class GPR:
    def __init__(self):
        self.kernel = GPy.kern.RBF(input_dim=1, variance=1.0, lengthscale=1.0)

    def fit(self, X, y):
        self.m = GPy.models.GPRegression(X, y, self.kernel)
        self.m.optimize(messages=True)

    def predict(self, X):
        return self.m.predict(X)


optimizer = {
    "Bayesian": {"gpr": GPR()},
}

hyper = Hyperactive(X, y)
hyper.add_search(model, search_space, optimizer=optimizer, n_iter=20)
hyper.run()
