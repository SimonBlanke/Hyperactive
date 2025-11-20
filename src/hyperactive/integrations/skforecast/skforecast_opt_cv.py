# copyright: hyperactive developers, MIT License (see LICENSE file)

import copy
import numpy as np
from sklearn.base import BaseEstimator

from hyperactive.experiment.integrations.skforecast_forecasting import (
    SkforecastExperiment,
)


class SkforecastOptCV(BaseEstimator):
    """Tune a skforecast forecaster via any optimizer in the hyperactive toolbox.

    Parameters
    ----------
    forecaster : skforecast forecaster
        The forecaster to tune.

    optimizer : hyperactive BaseOptimizer
        The optimizer to be used for hyperparameter search.

    steps : int
        Number of steps to predict

    metric : str or callable
        Metric used to quantify the goodness of fit of the model

    initial_train_size : int
        Number of samples in the initial training set

    exog : pandas Series or DataFrame, optional
        Exogenous variable/s used in the evaluation experiment

    refit : bool, optional
        Whether to re-fit the forecaster in each iteration

    fixed_train_size : bool, optional
        If True, the train size doesn't increase but moves by `steps` in each iteration

    gap : int, optional
        Number of samples to exclude from the end of each training set and the start of the test set

    allow_incomplete_fold : bool, optional
        If True, the last fold is allowed to have fewer samples than `steps`

    return_best : bool, optional
        If True, the best model is returned

    n_jobs : int or 'auto', optional
        Number of jobs to run in parallel

    verbose : bool, optional
        Print summary figures

    show_progress : bool, optional
        Whether to show a progress bar
    """

    def __init__(
        self,
        forecaster,
        optimizer,
        steps,
        metric,
        initial_train_size,
        exog=None,
        refit=False,
        fixed_train_size=False,
        gap=0,
        allow_incomplete_fold=True,
        return_best=False,
        n_jobs="auto",
        verbose=False,
        show_progress=False,
    ):
        self.forecaster = forecaster
        self.optimizer = optimizer
        self.steps = steps
        self.metric = metric
        self.initial_train_size = initial_train_size
        self.exog = exog
        self.refit = refit
        self.fixed_train_size = fixed_train_size
        self.gap = gap
        self.allow_incomplete_fold = allow_incomplete_fold
        self.return_best = return_best
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.show_progress = show_progress

    def fit(self, y, exog=None):
        """Fit to training data.

        Parameters
        ----------
        y : pandas Series
            Target time series to which to fit the forecaster.
        exog : pandas Series or DataFrame, optional
            Exogenous variables.

        Returns
        -------
        self : returns an instance of self.
        """
        current_exog = exog if exog is not None else self.exog

        experiment = SkforecastExperiment(
            forecaster=self.forecaster,
            y=y,
            steps=self.steps,
            metric=self.metric,
            initial_train_size=self.initial_train_size,
            exog=current_exog,
            refit=self.refit,
            fixed_train_size=self.fixed_train_size,
            gap=self.gap,
            allow_incomplete_fold=self.allow_incomplete_fold,
            return_best=self.return_best,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )

        if hasattr(self.optimizer, "clone"):
            optimizer = self.optimizer.clone()
        else:
            optimizer = copy.deepcopy(self.optimizer)

        optimizer.set_params(experiment=experiment)
        best_params = optimizer.solve()

        self.best_params_ = best_params
        self.best_forecaster_ = copy.deepcopy(self.forecaster)
        self.best_forecaster_.set_params(best_params)

        # Refit model with best parameters on the whole dataset
        self.best_forecaster_.fit(y=y, exog=current_exog)

        return self

    def predict(self, steps, exog=None, **kwargs):
        """Forecast time series at future horizon.

        Parameters
        ----------
        steps : int
            Number of steps to predict.
        exog : pandas Series or DataFrame, optional
            Exogenous variables.

        Returns
        -------
        predictions : pandas Series
            Predicted values.
        """
        return self.best_forecaster_.predict(steps=steps, exog=exog, **kwargs)
