import numpy as np

from hyperactive import BaseExperiment


class SphereFunction(BaseExperiment):
    def __init__(self, const, n_dim=2):
        super().__init__()

        self.const = const
        self.n_dim = n_dim

    def _score(self, **params):
        return np.sum(np.array(params) ** 2) + self.const


class AckleyFunction(BaseExperiment):
    def __init__(self, A):
        super().__init__()
        self.A = A

    def _score(self, **params):
        x = params["x0"]
        y = params["x1"]

        loss1 = -self.A * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
        loss2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        loss3 = np.exp(1)
        loss4 = self.A

        return -(loss1 + loss2 + loss3 + loss4)
