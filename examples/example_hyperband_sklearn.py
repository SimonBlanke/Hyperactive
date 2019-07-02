from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from hyperactive import RandomSearchOptimizer

breast_cancer_data = load_breast_cancer()
X = breast_cancer_data.data
y = breast_cancer_data.target

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

Optimizer = RandomSearchOptimizer(
    search_config, n_iter=100, n_jobs=1, hyperband_init=4, verbosity=0
)

# search best hyperparameter for given data
Optimizer.fit(X_train, y_train)

# predict from test data
prediction = Optimizer.predict(X_test)

# calculate accuracy score
score = Optimizer.score(X_test, y_test)
