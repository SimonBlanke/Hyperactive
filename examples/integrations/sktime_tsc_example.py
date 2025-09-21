# 1. defining the tuned estimator:
from sktime.classification.dummy import DummyClassifier
from sklearn.model_selection import KFold
from hyperactive.integrations.sktime import TSCOptCV
from hyperactive.opt import GridSearchSk as GridSearch

param_grid = {"strategy": ["most_frequent", "stratified"]}
tuned_naive = TSCOptCV(
    DummyClassifier(),
    GridSearch(param_grid),
    cv=KFold(n_splits=2, shuffle=False),
)

# 2. fitting the tuned estimator:
from sktime.datasets import load_unit_test
X_train, y_train = load_unit_test(
    return_X_y=True, split="TRAIN", return_type="pd-multiindex"
)
X_test, _ = load_unit_test(
    return_X_y=True, split="TEST", return_type="pd-multiindex"
)

tuned_naive.fit(X_train, y_train)

# 3. obtaining predictions
y_pred = tuned_naive.predict(X_test)

# 4. obtaining best parameters and best estimator
best_params = tuned_naive.best_params_
best_classifier = tuned_naive.best_estimator_
