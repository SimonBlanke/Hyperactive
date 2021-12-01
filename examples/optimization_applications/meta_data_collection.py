import pandas as pd
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive

data = load_iris()
X, y = data.data, data.target


def model1(opt):
    knr = KNeighborsClassifier(n_neighbors=opt["n_neighbors"])
    scores = cross_val_score(knr, X, y, cv=10)
    score = scores.mean()

    return score


search_space = {"n_neighbors": list(range(1, 50)), "leaf_size": list(range(5, 60, 5))}


hyper = Hyperactive()
hyper.add_search(model1, search_space, n_iter=500, memory=True)
hyper.run()

search_data = hyper.search_data(model1)
# save the search data of a model for later use
search_data.to_csv("./model1.csv", index=False)


# load the search data and pass it to "memory_warm_start"
search_data_loaded = pd.read_csv("./model1.csv")

hyper = Hyperactive()
hyper.add_search(
    model1, search_space, n_iter=500, memory=True, memory_warm_start=search_data_loaded
)
hyper.run()
