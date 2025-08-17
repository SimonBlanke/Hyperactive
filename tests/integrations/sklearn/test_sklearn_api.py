"""Test module for sklearn API integration."""

import numpy as np
import pytest
from sklearn import datasets, svm
from sklearn.decomposition import PCA
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted

from hyperactive.integrations import HyperactiveSearchCV
from hyperactive.optimizers import RandomSearchOptimizer

iris = datasets.load_iris()
X, y = iris.data, iris.target

nb = GaussianNB()
svc = svm.SVC()
pca = PCA()


nb_params = {
    "var_smoothing": [1e-9, 1e-8],
}
svc_params = {"kernel": ["linear", "rbf"], "C": [1, 10]}
pca_params = {
    "n_components": [2, 3],
}


opt = RandomSearchOptimizer()


def test_fit():
    """Test fitting the HyperactiveSearchCV estimator."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    check_is_fitted(search)


def test_not_fitted():
    """Test behavior when estimator is not fitted."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    assert not search.fit_successful

    with pytest.raises(NotFittedError):
        check_is_fitted(search)

    assert not search.fit_successful


def test_false_params():
    """Test error handling with invalid parameters."""
    search = HyperactiveSearchCV(svc, nb_params, opt)
    with pytest.raises(ValueError):
        search.fit(X, y)

    assert not search.fit_successful


def test_score():
    """Test scoring functionality of the fitted estimator."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)
    score = search.score(X, y)

    assert isinstance(score, float)


def test_classes_():
    """Test access to fitted classes."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    assert [0, 1, 2] == list(search.classes_)


def test_score_samples():
    """Test score_samples method raises AttributeError."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.score_samples(X)


def test_predict():
    """Test prediction functionality."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)
    result = search.predict(X)

    assert isinstance(result, np.ndarray)


def test_predict_proba():
    """Test predict_proba method behavior."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.predict_proba(X)

    search = HyperactiveSearchCV(nb, nb_params, opt)
    search.fit(X, y)
    result = search.predict(X)

    assert isinstance(result, np.ndarray)


def test_predict_log_proba():
    """Test predict_log_proba method behavior."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.predict_log_proba(X)

    search = HyperactiveSearchCV(nb, nb_params, opt)
    search.fit(X, y)
    result = search.predict_log_proba(X)

    assert isinstance(result, np.ndarray)


def test_decision_function():
    """Test decision_function method."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)
    result = search.decision_function(X)

    assert isinstance(result, np.ndarray)


def test_transform():
    """Test transform method behavior."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.transform(X)

    search = HyperactiveSearchCV(pca, pca_params, opt)
    search.fit(X, y)
    result = search.transform(X)

    assert isinstance(result, np.ndarray)


def test_inverse_transform():
    """Test inverse_transform method behavior."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    with pytest.raises(AttributeError):
        search.inverse_transform(X)

    search = HyperactiveSearchCV(pca, pca_params, opt)
    search.fit(X, y)
    result = search.inverse_transform(search.transform(X))

    assert isinstance(result, np.ndarray)


def test_best_params_and_score():
    """Test access to best parameters and score."""
    search = HyperactiveSearchCV(svc, svc_params, opt)
    search.fit(X, y)

    best_params = search.best_params_
    best_score = search.best_score_

    assert "kernel" in best_params and "C" in best_params
    assert isinstance(best_score, float)


def test_search_data():
    """Test access to search data after optimization."""
    n_iter = 10
    search = HyperactiveSearchCV(svc, svc_params, opt, n_iter=n_iter)
    search.fit(X, y)

    search_data = search.search_data_
    columns = search_data.columns

    assert len(search_data) == n_iter
    assert "kernel" in columns and "C" in columns
