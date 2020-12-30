import numpy as np


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive, BayesianOptimizer

data = load_iris()
X, y = data.data, data.target


def model(opt):
    knr = KNeighborsClassifier(n_neighbors=opt["n_neighbors"])
    scores = cross_val_score(knr, X, y, cv=5)
    score = scores.mean()

    return score


search_space = {
    "n_neighbors": list(range(1, 100)),
}


hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=100)
hyper.run()

results = hyper.results(model)
values = results[list(search_space.keys())].values
scores = results["score"].values

warm_start_smbo = (values, scores)

optimizer = BayesianOptimizer(
    xi=0.03, warm_start_smbo=warm_start_smbo, rand_rest_p=0.1
)

hyper = Hyperactive()
hyper.add_search(model, search_space, optimizer=optimizer, n_iter=100)
hyper.run()

