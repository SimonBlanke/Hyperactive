# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


# NOTE Implementations of following methods from:
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/model_selection/_search.py
# Tag: 1.5.1


def _check_refit(search_cv, attr):
    if not search_cv.refit:
        raise AttributeError(
            f"This {type(search_cv).__name__} instance was initialized with "
            f"`refit=False`. {attr} is available only after refitting on the best "
            "parameters. You can refit an estimator manually using the "
            "`best_params_` attribute"
        )


def _estimator_has(attr):
    def check(self):
        _check_refit(self, attr)
        if hasattr(self, "best_estimator_"):
            # raise an AttributeError if `attr` does not exist
            getattr(self.best_estimator_, attr)
            return True
        # raise an AttributeError if `attr` does not exist
        getattr(self.estimator, attr)
        return True

    return check
