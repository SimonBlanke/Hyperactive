from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive

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


"""
- set memory to "long" to enable Hyperactive to collect and save data about the search
"""
hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=100, memory="long")
hyper.run()


"""
- when starting a new search Hyperactive will load the previously saved search data
- loaded search data will be used to save computation time
"""
hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=100, memory="long")
hyper.run()
