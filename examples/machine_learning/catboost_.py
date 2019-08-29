import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# this defines the model and hyperparameter search space
search_config = {
    "catboost.CatBoostClassifier": {
        "iterations": [3],
        "learning_rate": np.arange(0.01, 0.1, 0.01),
        "depth": range(2, 20),
        "verbose": [0],
        "thread_count": [1],
    }
}

opt = RandomSearchOptimizer(search_config, n_iter=10, n_jobs=4, cv=3)

# search best hyperparameter for given data
opt.fit(X, y)

# predict from test data
prediction = opt.predict(X_test)

# calculate score
score = opt.score(X_test, y_test)

print("\ntest score of best model:", score)
