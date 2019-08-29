import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hyperactive import EvolutionStrategyOptimizer

cancer_data = load_breast_cancer()
X = cancer_data.data
y = cancer_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

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

opt = EvolutionStrategyOptimizer(search_config, n_iter=10, n_jobs=4)

# search best hyperparameter for given data
opt.fit(X_train, y_train)

# predict from test data
prediction = opt.predict(X_test)

# calculate score
score = opt.score(X_test, y_test)

print("\ntest score of best model:", score)
