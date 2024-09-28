import pytest
import numpy as np

from sklearn import svm, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.isotonic import IsotonicRegression
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError

from sklearn.utils.validation import check_is_fitted

from hyperactive.integrations import HyperactiveSearchCV
from hyperactive.optimizers import RandomSearchOptimizer


iris = datasets.load_iris()
X, y = iris.data, iris.target


ir = IsotonicRegression()
nb = GaussianNB()
svc = svm.SVC()
pca = PCA(n_components=2)


nb_params = {
    "var_smoothing": [1e-9, 1e-8],
}
svc_params = {"kernel": ["linear", "rbf"], "C": [1, 10]}
pca_params = {
    "n_components": [2, 3],
}


opt = RandomSearchOptimizer()


def test_fit():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    check_is_fitted(search)


def test_not_fitted():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    assert not search.fit_successful

    with pytest.raises(NotFittedError):
        check_is_fitted(search)

    assert not search.fit_successful


def test_false_params():
    search = HyperactiveSearchCV(svc, nb_params, opt)
    with pytest.raises(ValueError):
        search.fit(X, y)

    assert not search.fit_successful


def test_score():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)
    score = search.score(X, y)

    assert isinstance(score, float)


def test_classes_():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    assert [0, 1, 2] == list(search.classes_)


def test_score_samples():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.score_samples(X)


def test_predict():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)
    result = search.predict(X)

    assert isinstance(result, np.ndarray)


def test_predict_proba():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.predict_proba(X)

    search = HyperactiveSearchCV(nb, nb_params, opt)
    search.fit(X, y)
    result = search.predict(X)

    assert isinstance(result, np.ndarray)


def test_predict_log_proba():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.predict_log_proba(X)

    search = HyperactiveSearchCV(nb, nb_params, opt)
    search.fit(X, y)
    result = search.predict_log_proba(X)

    assert isinstance(result, np.ndarray)


def test_decision_function():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)
    result = search.decision_function(X)

    assert isinstance(result, np.ndarray)


def test_transform():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.transform(X)

    search = HyperactiveSearchCV(pca, pca_params, opt)
    search.fit(X, y)
    result = search.transform(X)

    assert isinstance(result, np.ndarray)


def test_inverse_transform():
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.inverse_transform(X)

    search = HyperactiveSearchCV(pca, pca_params, opt)
    search.fit(X, y)
    result = search.inverse_transform(search.transform(X))

    assert isinstance(result, np.ndarray)
