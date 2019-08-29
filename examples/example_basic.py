from sklearn.datasets import load_iris

from hyperactive import RandomSearchOptimizer

iris_data = load_iris()
X, y = iris_data.data, iris_data.target

search_config = {
    "sklearn.ensemble.RandomForestClassifier": {"n_estimators": range(10, 100, 10)}
}

opt = RandomSearchOptimizer(search_config, n_iter=10)
opt.fit(X, y)
