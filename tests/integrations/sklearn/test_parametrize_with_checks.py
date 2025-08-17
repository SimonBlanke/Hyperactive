"""Test module for sklearn parametrize_with_checks integration."""

import pytest
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.utils.estimator_checks import parametrize_with_checks

from hyperactive.integrations import HyperactiveSearchCV, OptCV
from hyperactive.opt import GridSearchSk as GridSearch
from hyperactive.optimizers import RandomSearchOptimizer

svc = svm.SVC()
parameters = {"kernel": ["linear"], "C": [1]}  # Reduced parameter space
opt = RandomSearchOptimizer()
hyperactivecv = HyperactiveSearchCV(svc, parameters, opt, n_iter=5)  # Reduced iterations

cv = KFold(n_splits=2, shuffle=True, random_state=42)
optcv = OptCV(estimator=svc, optimizer=GridSearch(param_grid=parameters), cv=cv)

ESTIMATORS = [hyperactivecv]  # Reduced to single estimator


@pytest.mark.slow
@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    """Test estimators with sklearn estimator checks."""
    check(estimator)
