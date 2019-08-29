from sklearn.datasets import load_iris
from hyperactive import ParticleSwarmOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.linear_model.LogisticRegression": {
        "penalty": ["l1", "l2"],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0],
        "dual": [False],
        "solver": ["liblinear"],
        "multi_class": ["auto", "ovr"],
        "max_iter": range(300, 1000, 10),
    }
}

opt = ParticleSwarmOptimizer(search_config, n_iter=100, n_jobs=2, cv=5)

# search best hyperparameter for given data
opt.fit(X, y)
