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
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from hyperactive import Hyperactive, EvolutionStrategyOptimizer

data = load_boston()
X, y = data.data, data.target


# helper function that returns the selected training data features by index
def get_feature_indices(opt):
    feature_indices = []
    for key in opt.keys():
        if "feature" not in key:
            continue
        if opt[key] is False:
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


# each feature is used for training (True) or not used for training (False)
search_space = {
    "n_neighbors": list(range(1, 100)),
    "feature.0": [True, False],
    "feature.1": [True, False],
    "feature.2": [True, False],
    "feature.3": [True, False],
    "feature.4": [True, False],
    "feature.5": [True, False],
    "feature.6": [True, False],
    "feature.7": [True, False],
    "feature.8": [True, False],
    "feature.9": [True, False],
    "feature.10": [True, False],
    "feature.11": [True, False],
    "feature.12": [True, False],
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
