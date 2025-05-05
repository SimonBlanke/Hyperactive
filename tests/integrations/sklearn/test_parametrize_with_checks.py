from sklearn import svm

from hyperactive.integrations import HyperactiveSearchCV, OptCV
from hyperactive.opt import GridSearch
from hyperactive.optimizers import RandomSearchOptimizer

from sklearn.utils.estimator_checks import parametrize_with_checks


svc = svm.SVC()
parameters = {"kernel": ["linear", "rbf"], "C": [1, 10]}
opt = RandomSearchOptimizer()
hyperactivecv = HyperactiveSearchCV(svc, parameters, opt)

optcv = OptCV(estimator=svc, optimizer=GridSearch(parameters))

ESTIMATORS = [hyperactivecv, optcv]


@parametrize_with_checks(ESTIMATORS)
def test_estimators(estimator, check):
    check(estimator)
