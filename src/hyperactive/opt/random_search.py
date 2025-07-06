"""Grid search optimizer."""

# copyright: hyperactive developers, MIT License (see LICENSE file)

from collections.abc import Sequence

import numpy as np

from sklearn.model_selection import ParameterSampler

from hyperactive.base import BaseOptimizer


class RandomSearchSk(BaseOptimizer):
    """Random search optimizer leveraging sklearn's ``ParameterSampler``.

    Parameters
    ----------
    param_distributions : dict[str, list | scipy.stats.rv_frozen]
        Search space specification. Discrete lists are sampled uniformly;
        scipy distribution objects are sampled via their ``rvs`` method.
    n_iter : int, default=10
        Number of parameter sets to evaluate.
    random_state : int | np.random.RandomState | None, default=None
        Controls the pseudo-random generator for reproducibility.
    error_score : float, default=np.nan
        Score assigned when the experiment raises an exception.
    experiment : BaseExperiment, optional
        Callable returning a scalar score when invoked with keyword
        arguments matching a parameter set.

    Attributes
    ----------
    best_params_ : dict[str, Any]
        Hyper-parameter configuration with the best (lowest) score.
    best_score_ : float
        Score achieved by ``best_params_``.
    best_index_ : int
        Index of ``best_params_`` in the sampled sequence.
    """

    def __init__(
        self,
        param_distributions=None,
        n_iter=10,
        random_state=None,
        error_score=np.nan,
        experiment=None,
    ):
        self.experiment = experiment
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state
        self.error_score = error_score

        super().__init__()
