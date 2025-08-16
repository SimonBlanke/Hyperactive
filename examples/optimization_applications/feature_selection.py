"""
This example shows how to select the best features for a model
and dataset.

The boston dataset has 13 features, therefore we have 13 search space
dimensions for the feature selection.

The function "get_feature_indices" returns the list of features that
where selected. This can be used to select the subset of features in "x_new".
"""

import numpy as np
import itertools
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from hyperactive import Hyperactive
from hyperactive.optimizers import EvolutionStrategyOptimizer


data = load_diabetes()
X, y = data.data, data.target


# helper function that returns the selected training data features by index
def get_feature_indices(opt):
    feature_indices = []
    for key in opt.keys():
        if "feature" not in key:
            continue
        if opt[key] == 0:
            continue

        nth_feature = int(key.rsplit(".", 1)[1])
        feature_indices.append(nth_feature)

    return feature_indices


def model(opt):
    feature_indices = get_feature_indices(opt)
    if len(feature_indices) == 0:
        return 0

    feature_idx_list = [idx for idx in feature_indices if idx is not None]
    x_new = X[:, feature_idx_list]

    knr = KNeighborsRegressor(n_neighbors=opt["n_neighbors"])
    scores = cross_val_score(knr, x_new, y, cv=5)
    score = scores.mean()

    return score


# each feature is used for training (1) or not used for training (0)
search_space = {
    "n_neighbors": list(range(1, 100)),
    "feature.0": [1, 0],
    "feature.1": [1, 0],
    "feature.2": [1, 0],
    "feature.3": [1, 0],
    "feature.4": [1, 0],
    "feature.5": [1, 0],
    "feature.6": [1, 0],
    "feature.7": [1, 0],
    "feature.8": [1, 0],
    "feature.9": [1, 0],
}


optimizer = EvolutionStrategyOptimizer(rand_rest_p=0.20)

hyper = Hyperactive()
hyper.add_search(
    model,
    search_space,
    n_iter=200,
    initialize={"random": 15},
    optimizer=optimizer,
)
hyper.run()
