"""Test module for sklearn parametrize_with_checks integration."""

from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from hyperactive.integrations import OptCV
from hyperactive.opt import GridSearchSk as GridSearch

svc = svm.SVC()
parameters = {"kernel": ["linear", "rbf"], "C": [1, 10]}

cv = KFold(n_splits=2, shuffle=True, random_state=42)
optcv = OptCV(estimator=svc, optimizer=GridSearch(param_grid=parameters), cv=cv)

ESTIMATORS = [optcv]


@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    """Test estimators with sklearn estimator checks."""
    check(estimator)
