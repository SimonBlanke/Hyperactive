import random
import numpy as pd
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive


def model(opt):
    knr = KNeighborsClassifier(n_neighbors=opt["n_neighbors"])
    scores = cross_val_score(knr, X, y, cv=5)
    score = scores.mean()

    return score


search_space = {
    "n_neighbors": list(range(1, 80)),
}


search_data_list = []

for i in range(25):
    n_samples = random.randint(100, 1000)
    n_features = random.randint(3, 20)
    n_informative = n_features - random.randint(0, n_features - 2)

    X, y = make_classification(
        n_samples=n_samples,
        n_classes=2,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        random_state=i,
    )

    hyper = Hyperactive(verbosity=False)
    hyper.add_search(model, search_space, n_iter=10)
    hyper.run()

    search_data = hyper.search_data(model)

    search_data["size_X"] = X.size
    search_data["itemsize_X"] = X.itemsize
    search_data["ndim_X"] = X.ndim

    search_data["size_y"] = y.size
    search_data["itemsize_y"] = y.itemsize
    search_data["ndim_y"] = y.ndim

    search_data_list.append(search_data)


meta_data = pd.concat(search_data_list)

X_meta = meta_data.drop(["score"], axis=1)
y_meta = meta_data["score"]


gbr = GradientBoostingRegressor()
gbr.fit(X_meta, y_meta)

data = load_iris()
X_new, y_new = data.data, data.target

X_meta_test = pd.DataFrame(range(1, 100), columns=["n_neighbors"])

X_meta_test["size_X"] = X_new.size
X_meta_test["itemsize_X"] = X_new.itemsize
X_meta_test["ndim_X"] = X_new.ndim

X_meta_test["size_y"] = y_new.size
X_meta_test["itemsize_y"] = y_new.itemsize
X_meta_test["ndim_y"] = y_new.ndim


y_meta_pred = gbr.predict(X_meta_test)

y_meta_pred_max_idx = y_meta_pred.argmax()
n_neighbors_best = search_space["n_neighbors"][y_meta_pred_max_idx]

hyper = Hyperactive()
hyper.add_search(model, search_space, n_iter=200)
hyper.run()
