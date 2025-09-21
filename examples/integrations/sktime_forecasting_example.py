# 1. defining the tuned estimator:
from sktime.forecasting.naive import NaiveForecaster
from sktime.split import ExpandingWindowSplitter
from hyperactive.integrations.sktime import ForecastingOptCV
from hyperactive.opt import GridSearchSk as GridSearch

param_grid = {"strategy": ["mean", "last", "drift"]}
tuned_naive = ForecastingOptCV(
    NaiveForecaster(),
    GridSearch(param_grid),
    cv=ExpandingWindowSplitter(
        initial_window=12, step_length=3, fh=range(1, 13)
    ),
)

# 2. fitting the tuned estimator:
from sktime.datasets import load_airline
from sktime.split import temporal_train_test_split
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=12)

tuned_naive.fit(y_train, fh=range(1, 13))

# 3. obtaining predictions
y_pred = tuned_naive.predict()

# 4. obtaining best parameters and best forecaster
best_params = tuned_naive.best_params_
best_forecaster = tuned_naive.best_forecaster_
