import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from hyperactive.search_config import SearchConfig
from hyperactive.optimizers import HillClimbingOptimizer

from hyperactive.base import BaseExperiment


class SklearnExperiment(BaseExperiment):
    def setup(self, estimator, X, y, cv=5):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.cv = cv

    def _score(self, params):
        model = self.estimator(**params)
        scores = cross_val_score(model, self.X, self.y, cv=self.cv)
        return scores.mean()


data = load_diabetes()
X, y = data.data, data.target


search_config = SearchConfig(
    max_depth=list(np.arange(2, 15, 1)),
    min_samples_split=list(np.arange(2, 25, 2)),
)

experiment = SklearnExperiment()
experiment.setup(DecisionTreeRegressor, X, y, cv=4)

optimizer1 = HillClimbingOptimizer()
optimizer1.add_search(experiment, search_config, n_iter=100, n_jobs=2)
optimizer1.run()
