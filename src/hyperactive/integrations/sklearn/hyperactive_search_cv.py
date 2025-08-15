# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from collections.abc import Callable
from typing import Dict, Type, Union

from sklearn.base import BaseEstimator, clone
from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.metrics import check_scoring

from hyperactive import Hyperactive
from hyperactive.experiment.integrations.sklearn_cv import SklearnCvExperiment

from ...optimizers import RandomSearchOptimizer
from ._compat import _check_method_params, _safe_refit, _safe_validate_X_y
from .best_estimator import BestEstimator as _BestEstimator_
from .checks import Checks


class HyperactiveSearchCV(BaseEstimator, _BestEstimator_, Checks):
    """
    HyperactiveSearchCV class for hyperparameter tuning using cross-validation with sklearn estimators.

    Parameters
    ----------
    - estimator: SklearnBaseEstimator
        The estimator to be tuned.
    - params_config: Dict[str, list]
        Dictionary containing the hyperparameter search space.
    - optimizer: Union[str, Type[RandomSearchOptimizer]], optional
        The optimizer to be used for hyperparameter search, default is "default".
    - n_iter: int, optional
        Number of parameter settings that are sampled, default is 100.
    - scoring: Callable | str | None, optional
        Scoring method to evaluate the predictions on the test set.
    - n_jobs: int, optional
        Number of jobs to run in parallel, default is 1.
    - random_state: int | None, optional
        Random seed for reproducibility.
    - refit: bool, optional
        Refit the best estimator with the entire dataset, default is True.
    - cv: int | "BaseCrossValidator" | Iterable | None, optional
        Determines the cross-validation splitting strategy.

    Methods
    -------
    - fit(X, y, **fit_params)
        Fit the estimator and tune hyperparameters.
    - score(X, y, **params)
        Return the score of the best estimator on the input data.
    """

    _required_parameters = ["estimator", "optimizer", "params_config"]

    def __init__(
        self,
        estimator: "SklearnBaseEstimator",
        params_config: Dict[str, list],
        optimizer: Union[str, Type[RandomSearchOptimizer]] = "default",
        n_iter: int = 100,
        *,
        scoring: Union[Callable, str, None] = None,
        n_jobs: int = 1,
        random_state: Union[int, None] = None,
        refit: bool = True,
        cv=None,
    ):
        super().__init__()

        self.estimator = estimator
        self.params_config = params_config
        self.optimizer = optimizer
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.refit = refit
        self.cv = cv

    def _refit(self, X, y=None, **fit_params):
        self.best_estimator_ = clone(self.estimator).set_params(
            **clone(self.best_params_, safe=False)
        )

        self.best_estimator_.fit(X, y, **fit_params)
        return self

    def _check_data(self, X, y):
        return _safe_validate_X_y(self, X, y)

    @Checks.verify_fit
    def fit(self, X, y, **fit_params):
        """
        Fit the estimator using the provided training data.

        Parameters
        ----------
        - X: array-like or sparse matrix, shape (n_samples, n_features)
            The training input samples.
        - y: array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        - **fit_params: dict of string -> object
            Additional fit parameters.

        Returns
        -------
        - self: object
            Returns the instance itself.
        """
        X, y = self._check_data(X, y)

        fit_params = _check_method_params(X, params=fit_params)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        experiment = SklearnCvExperiment(
            estimator=self.estimator,
            scoring=self.scorer_,
            cv=self.cv,
            X=X,
            y=y,
        )
        objective_function = experiment.score

        hyper = Hyperactive(verbosity=False)
        hyper.add_search(
            objective_function,
            search_space=self.params_config,
            optimizer=self.optimizer,
            n_iter=self.n_iter,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        hyper.run()

        self.best_params_ = hyper.best_para(objective_function)
        self.best_score_ = hyper.best_score(objective_function)
        self.search_data_ = hyper.search_data(objective_function)

        _safe_refit(self, X, y, fit_params)

        return self

    def score(self, X, y=None, **params):
        """
        Calculate the score of the best estimator on the input data.

        Parameters
        ----------
        - X: array-like or sparse matrix of shape (n_samples, n_features)
            The input samples.
        - y: array-like of shape (n_samples,), default=None
            The target values.
        - **params: dict
            Additional parameters to be passed to the scoring function.

        Returns
        -------
        - float
            The score of the best estimator on the input data.
        """
        return self.scorer_(self.best_estimator_, X, y, **params)

    @property
    def fit_successful(self):
        self._fit_successful
