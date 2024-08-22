# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from sklearn.model_selection import cross_validate
from sklearn.utils.validation import _num_samples


class ObjectiveFunctionAdapter:
    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def add_dataset(self, X, y):
        self.X = X
        self.y = y

    def add_validation(self, scoring, cv):
        self.scoring = scoring
        self.cv = cv

    def objective_function(self, params):
        cv_results = cross_validate(
            self.estimator,
            self.X,
            self.y,
            cv=self.cv,
        )

        add_info_d = {
            "score_time": cv_results["score_time"],
            "fit_time": cv_results["fit_time"],
            "n_test_samples": _num_samples(self.X),
        }

        return cv_results["test_score"].mean(), add_info_d
