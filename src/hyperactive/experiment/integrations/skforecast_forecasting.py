"""Experiment adapter for skforecast backtesting experiments."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import copy
import numpy as np
from hyperactive.base import BaseExperiment


class SkforecastExperiment(BaseExperiment):
    """Experiment adapter for skforecast backtesting experiments.

    This class is used to perform backtesting experiments using a given
    skforecast forecaster. It allows for hyperparameter tuning and evaluation of
    the model's performance.

    Parameters
    ----------
    forecaster : skforecast forecaster
        skforecast forecaster to benchmark

    y : pandas Series
        Target time series used in the evaluation experiment

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
        y,
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
        self.y = y
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

        super().__init__()

    def _evaluate(self, params):
        """Evaluate the parameters.

        Parameters
        ----------
        params : dict with string keys
            Parameters to evaluate.

        Returns
        -------
        float
            The value of the parameters as per evaluation.
        dict
            Additional metadata about the search.
        """
        from skforecast.model_selection import backtesting_forecaster
        from skforecast.model_selection import TimeSeriesFold

        forecaster = copy.deepcopy(self.forecaster)
        forecaster.set_params(params)

        cv = TimeSeriesFold(
            steps=self.steps,
            initial_train_size=self.initial_train_size,
            refit=self.refit,
            fixed_train_size=self.fixed_train_size,
            gap=self.gap,
            allow_incomplete_fold=self.allow_incomplete_fold,
        )

        results, _ = backtesting_forecaster(
            forecaster=forecaster,
            y=self.y,
            cv=cv,
            metric=self.metric,
            exog=self.exog,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )

        if isinstance(self.metric, str):
            metric_name = self.metric
        else:
            metric_name = (
                self.metric.__name__ if hasattr(self.metric, "__name__") else "score"
            )

        # backtesting_forecaster returns a DataFrame
        res_float = results[metric_name].iloc[0]

        return res_float, {"results": results}
