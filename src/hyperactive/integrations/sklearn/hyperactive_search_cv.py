# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.base import BaseEstimator, clone
from sklearn.metrics import check_scoring
from sklearn.utils.validation import indexable, _check_method_params


from hyperactive import Hyperactive

from .objective_function_adapter import ObjectiveFunctionAdapter


class HyperactiveSearchCV(BaseEstimator):
    _required_parameters = ["estimator", "optimizer", "params_config"]

    def __init__(
        self,
        estimator,
        optimizer,
        params_config,
        n_iter=100,
        *,
        scoring=None,
        n_jobs=1,
        random_state=None,
        refit=True,
        cv=None,
    ):
        self.estimator = estimator
        self.optimizer = optimizer
        self.params_config = params_config
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
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.fit(X, y, **fit_params)
        return self

    def fit(self, X, y, **params):
        X, y = indexable(X, y)
        X, y = self._validate_data(X, y)

        params = _check_method_params(X, params=params)
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        objective_function_adapter = ObjectiveFunctionAdapter(
            self.estimator,
        )
        objective_function_adapter.add_dataset(X, y)
        objective_function_adapter.add_validation(self.scorer_, self.cv)

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

        if self.refit:
            self._refit(X, y, **params)

        return self

    def score(self, X, y=None, **params):
        return self.scorer_(self.best_estimator_, X, y, **params)
