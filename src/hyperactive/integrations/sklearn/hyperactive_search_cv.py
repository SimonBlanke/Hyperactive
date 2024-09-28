# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from collections.abc import Iterable, Callable
from typing import Union, Dict, Type

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.utils.validation import indexable, _check_method_params

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.model_selection import BaseCrossValidator

from hyperactive import Hyperactive

from .objective_function_adapter import ObjectiveFunctionAdapter
from .best_estimator import BestEstimator as _BestEstimator_
from .checks import Checks
from ...optimizers import RandomSearchOptimizer


class HyperactiveSearchCV(BaseEstimator, _BestEstimator_, Checks):
    _required_parameters = ["estimator", "optimizer", "params_config"]

    def __init__(
        self,
        estimator: "SklearnBaseEstimator",
        params_config: Dict[str, list],
        optimizer: Union[str, Type[RandomSearchOptimizer]] = "default",
        n_iter: int = 100,
        *,
        scoring: Callable | str | None = None,
        n_jobs: int = 1,
        random_state: int | None = None,
        refit: bool = True,
        cv: int | "BaseCrossValidator" | Iterable | None = None,
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

    def _refit(
        self,
        X,
        y=None,
        **fit_params,
    ):
        self.best_estimator_ = clone(self.estimator).set_params(
            **self.best_params_
        )
        self.best_estimator_.fit(X, y, **fit_params)
        return self

    @Checks.verify_fit
    def fit(self, X, y, **fit_params):
        X, y = indexable(X, y)
        X, y = self._validate_data(X, y)

        fit_params = _check_method_params(X, params=fit_params)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        objective_function_adapter = ObjectiveFunctionAdapter(
            self.estimator,
        )
        objective_function_adapter.add_dataset(X, y)
        objective_function_adapter.add_validation(self.scorer_, self.cv)
        objective_function = objective_function_adapter.objective_function

        hyper = Hyperactive(verbosity=False)
        hyper.add_search(
            objective_function_adapter.objective_function,
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

        if self.refit:
            self._refit(X, y, **fit_params)

        return self

    def score(self, X, y=None, **params):
        return self.scorer_(self.best_estimator_, X, y, **params)
