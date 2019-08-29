from sklearn.datasets import load_iris
from hyperactive import RandomSearchOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 200, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
    }
}

"""
The memory will remember previous evaluations done during the optimization process.
Instead of retraining the model, it accesses the memory and uses the saved score/loss.
This shows as a speed up during the optimization process, since the whole search space has been explored.
"""
opt = RandomSearchOptimizer(search_config, n_iter=1000, memory=True)

# search best hyperparameter for given data
opt.fit(X, y)
