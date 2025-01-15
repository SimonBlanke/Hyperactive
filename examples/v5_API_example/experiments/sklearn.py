from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


from hyperactive import BaseExperiment


class SklearnExperiment(BaseExperiment):
    """
    Initializes the SklearnExperiment with the given estimator, data, and cross-validation settings.

    Parameters
    ----------
    estimator : object
        The machine learning estimator to be used for the experiment.
    X : array-like
        The input data for training the model.
    y : array-like
        The target values corresponding to the input data.
    cv : int, optional
        The number of cross-validation folds (default is 4).
    """

    def __init__(self, estimator, X, y, cv=4):
        super().__init__()

        self.estimator = estimator
        self.X = X
        self.y = y
        self.cv = cv

    def _score(self, **params):
        model = self.estimator(**params)
        scores = cross_val_score(model, self.X, self.y, cv=self.cv)
        return scores.mean()


class GradientBoostingExperiment(BaseExperiment):
    """
    A class for conducting experiments with Gradient Boosting Regressor using cross-validation.

    This class inherits from BaseExperiment and allows users to perform experiments
    with the GradientBoostingRegressor from sklearn. Users can specify the input
    features, target values, and the number of cross-validation folds.

    Attributes:
        estimator (type): The regression model to be used, default is GradientBoostingRegressor.
        X (array-like): The input features for the model.
        y (array-like): The target values for the model.
        cv (int): The number of cross-validation folds.

    Methods:
        _score(**params): Evaluates the model using cross-validation and returns the mean score.
    """

    def __init__(self, X, y, cv=4):
        super().__init__()

        self.estimator = GradientBoostingRegressor  # The user could also predefine the estimator
        self.X = X
        self.y = y
        self.cv = cv

    def _score(self, **params):
        model = self.estimator(**params)
        scores = cross_val_score(model, self.X, self.y, cv=self.cv)
        return scores.mean()
