from sklearn.datasets import load_iris

from hyperactive import RandomSearchOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

search_config = {
    "sklearn.ensemble.RandomForestClassifier": {"n_estimators": range(10, 100, 10)}
}

Optimizer = RandomSearchOptimizer(search_config, n_iter=10, verbosity=0)
Optimizer.fit(X, y)
