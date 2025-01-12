from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor


from hyperactive import BaseExperiment


class SklearnExperiment(BaseExperiment):
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
