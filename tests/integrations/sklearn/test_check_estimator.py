from sklearn import svm

from hyperactive.integrations import HyperactiveSearchCV
from hyperactive.optimizers import RandomSearchOptimizer

from sklearn.utils.estimator_checks import check_estimator


svc = svm.SVC()
parameters = {"kernel": ["linear", "rbf"], "C": [1, 10]}
opt = RandomSearchOptimizer()


def test_check_estimator():
    check_estimator(HyperactiveSearchCV(svc, opt, parameters))
