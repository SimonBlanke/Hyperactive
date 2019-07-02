import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from hyperactive import SimulatedAnnealingOptimizer

iris_data = load_iris()
X = iris_data.data
y = iris_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# this defines the model and hyperparameter search space
search_config = {
    "xgboost.XGBClassifier": {
        "n_estimators": range(30, 200, 10),
        "max_depth": range(1, 11),
        "learning_rate": [1e-3, 1e-2, 1e-1, 0.5, 1.0],
        "subsample": np.arange(0.05, 1.01, 0.05),
        "min_child_weight": range(1, 21),
        "nthread": [1],
    }
}

Optimizer = SimulatedAnnealingOptimizer(
    search_config, n_iter=20, n_jobs=4, verbosity=0
)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
