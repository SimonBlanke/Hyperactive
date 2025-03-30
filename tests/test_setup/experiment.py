import numpy as np

from hyperactive.experiment import BaseExperiment


class SphereFunction(BaseExperiment):
    def setup(self, n_dim=2):
        self.n_dim = n_dim

    def objective_function(self, params):
        return np.sum(params["x0"] ** 2 + params["x1"] ** 2)
