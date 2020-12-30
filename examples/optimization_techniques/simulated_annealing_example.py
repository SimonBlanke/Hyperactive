from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive, SimulatedAnnealingOptimizer

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


optimizer = SimulatedAnnealingOptimizer(
    epsilon=0.1,
    distribution="laplace",
    n_neighbours=4,
    rand_rest_p=0.1,
    p_accept=0.15,
    norm_factor="adaptive",
    annealing_rate=0.999,
    start_temp=0.8,
)

hyper = Hyperactive()
hyper.add_search(model, search_space, optimizer=optimizer, n_iter=100)
hyper.run()
