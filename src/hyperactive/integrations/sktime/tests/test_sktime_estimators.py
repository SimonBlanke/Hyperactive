"""Integration tests for sktime tuners."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

SKTIME_AVAILABLE = _check_soft_dependencies("sktime", severity="none")

if SKTIME_AVAILABLE:
    from hyperactive.integrations.sktime import ForecastingOptCV, TSCOptCV

    EST_TO_TEST = [ForecastingOptCV, TSCOptCV]
else:
    EST_TO_TEST = []

pytestmark = pytest.mark.skipif(
    not SKTIME_AVAILABLE, reason="sktime soft dependency not available"
)


@pytest.mark.parametrize("estimator", EST_TO_TEST)
def test_sktime_estimator(estimator):
    """Test sktime estimator via check_estimator."""
    from sktime.utils.estimator_checks import check_estimator

    check_estimator(estimator, raise_exceptions=True)
    # The above line collects all API conformance tests in sktime and runs them.
    # It will raise an error if the estimator is not API conformant.


def test_tune_by_instance_fallback_when_not_panel():
    """Ensure tune_by_instance gracefully falls back for univariate data."""
    import numpy as np
    import pandas as pd
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.split import SingleWindowSplitter

    from hyperactive.opt.gridsearch import GridSearchSk

    y = pd.Series(np.arange(24, dtype=float))
    fh = [1]
    splitter = SingleWindowSplitter(fh=fh, window_length=12)
    optimizer = GridSearchSk(param_grid={"window_length": [2, 4]})

    tuner = ForecastingOptCV(
        forecaster=NaiveForecaster(strategy="last"),
        optimizer=optimizer,
        cv=splitter,
        tune_by_instance=True,
    )

    tuned = tuner.fit(y, fh=fh)

    assert isinstance(tuned.best_params_, dict)
    assert tuned.best_index_ == 0
    assert not hasattr(tuned, "forecasters_")
    assert tuned.refit_time_ >= 0.0
