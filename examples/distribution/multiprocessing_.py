import numpy as np
from sklearn.datasets import load_breast_cancer

from hyperactive import RandomSearchOptimizer

breast_cancer_data = load_breast_cancer()
X = breast_cancer_data.data
y = breast_cancer_data.target

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(2, 21),
    },
    "sklearn.neighbors.KNeighborsClassifier": {
        "n_neighbors": range(1, 101),
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    },
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(10, 100, 10),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "max_depth": range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
        "subsample": list(np.arange(0.05, 1.01, 0.05)),
        "max_features": list(np.arange(0.05, 1.01, 0.05)),
    },
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    },
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=100, n_jobs=-1)

# search best hyperparameter for given data
Optimizer.fit(X, y)
