# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

label_encoder_dict = {
    "sklearn.neighbors.KNeighborsClassifier": {
        "weights": {"uniform": 0, "distance": 1},
        "algorithm": {"auto": 0, "ball_tree": 1, "kd_tree": 2, "brute": 3},
        "metric": {"minkowski": 0},
        "metric_params": {None: 0},
        "n_jobs": {None: 0},
    },
    "sklearn.ensemble.RandomForestClassifier": {
        "criterion": {"gini": 0, "entropy": 1},
        "max_depth": {None: 0},
        "max_features": {None: 0, "auto": 1},
        "max_leaf_nodes": {None: 0},
        "min_impurity_split": {None: 0},
        "n_jobs": {None: 0},
        "random_state": {None: 0},
        "class_weight": {None: 0, "balanced": 1},
    },
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": {"gini": 0, "entropy": 1},
        "splitter": {"best": 0, "random": 1},
        "max_features": {None: 0, "auto": 1},
        "class_weight": {None: 0, "balanced": 1},
        "random_state": {None: 0},
        "max_leaf_nodes": {None: 0},
        "min_impurity_split": {None: 0},
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "loss": {"deviance": 0, "exponential": 1},
        "criterion": {"friedman_mse": 0, "mse": 1, "mae": 2},
        "max_depth": {None: 0},
        "max_features": {None: 0, "auto": 1},
        "max_leaf_nodes": {None: 0},
        "min_impurity_split": {None: 0},
        "n_jobs": {None: 0},
        "n_iter_no_change": {None: 0},
        "init": {None: 0},
        "random_state": {None: 0},
        "class_weight": {None: 0, "balanced": 1},
        "presort": {"auto": 0},
    },
}
