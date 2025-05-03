from sklearn import svm

from hyperactive.integrations import HyperactiveSearchCV
from hyperactive.optimizers import RandomSearchOptimizer

from sklearn.utils.estimator_checks import parametrize_with_checks


svc = svm.SVC()
parameters = {"kernel": ["linear", "rbf"], "C": [1, 10]}
opt = RandomSearchOptimizer()


@parametrize_with_checks([HyperactiveSearchCV(svc, parameters, opt)])
def test_estimators(estimator, check):
    check(estimator)
