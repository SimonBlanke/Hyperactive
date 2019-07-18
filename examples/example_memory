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


Optimizer = RandomSearchOptimizer(
    search_config, n_iter=1000, n_jobs=1, memory=True, cv=3
)

# search best hyperparameter for given data
Optimizer.fit(X, y)
