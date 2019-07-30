# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

random_state = 0
cv = 2
n_jobs = 2

search_config = {
    "sklearn.tree.DecisionTreeClassifier": {
        "criterion": ["gini", "entropy"],
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}


def test_methods():
    from hyperactive import RandomSearchOptimizer

    Optimizer = RandomSearchOptimizer(search_config, n_iter=10, verbosity=0)
    Optimizer.fit(X_train, y_train)
    Optimizer.predict(X_test)
    Optimizer.score(X_test, y_test)
