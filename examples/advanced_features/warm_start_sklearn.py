from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import SimulatedAnnealingOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the model and hyperparameter search space
search_config = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": range(10, 100, 10),
        "max_depth": [3, 4, 5, 6],
        "criterion": ["gini", "entropy"],
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(2, 21),
    }
}

start_point = {
    "sklearn.ensemble.RandomForestClassifier.0": {
        "n_estimators": [30],
        "max_depth": [6],
        "criterion": ["entropy"],
        "min_samples_split": [12],
        "min_samples_leaf": [16],
    }
}

opt = SimulatedAnnealingOptimizer(
    search_config, n_iter=100, n_jobs=4, warm_start=start_point, verbosity=0
)

# search best hyperparameter for given data
opt.fit(X_train, y_train)

# predict from test data
prediction = opt.predict(X_test)

# calculate accuracy score
score = opt.score(X_test, y_test)
