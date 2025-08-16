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

    @staticmethod
    def _is_distribution(obj) -> bool:
        """Return True if *obj* looks like a scipy frozen distribution."""
        return callable(getattr(obj, "rvs", None))

    def _check_param_distributions(self, param_distributions):
        """Validate ``param_distributions`` similar to sklearn ≤1.0.x."""
        if hasattr(param_distributions, "items"):
            param_distributions = [param_distributions]

        for p in param_distributions:
            for name, v in p.items():
                if self._is_distribution(v):
                    # Assume scipy frozen distribution - nothing to check
                    continue

                if isinstance(v, np.ndarray) and v.ndim > 1:
                    raise ValueError("Parameter array should be one-dimensional.")

                if isinstance(v, str) or not isinstance(v, (np.ndarray, Sequence)):
                    raise ValueError(
                        f"Parameter distribution for ({name}) must be a list, numpy "
                        f"array, or scipy.stats ``rv_frozen``, but got ({type(v)})."
                        " Single values need to be wrapped in a sequence."
                    )

                if len(v) == 0:
                    raise ValueError(
                        f"Parameter values for ({name}) need to be a "
                        "non-empty sequence."
                    )

    def _solve(
        self,
        experiment,
        param_distributions,
        n_iter,
        random_state,
        error_score,
    ):
        """Sample ``n_iter`` points and return the best parameter set."""
        self._check_param_distributions(param_distributions)

        sampler = ParameterSampler(
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=random_state,
        )
        candidate_params = list(sampler)

        scores: list[float] = []
        for candidate_param in candidate_params:
            try:
                score = experiment(**candidate_param)
            except Exception:  # noqa: B904
                score = error_score
            scores.append(score)

        best_index = int(np.argmin(scores))  # lower-is-better convention
        best_params = candidate_params[best_index]

        # public attributes for external consumers
        self.best_index_ = best_index
        self.best_score_ = float(scores[best_index])
        self.best_params_ = best_params

        return best_params

    @classmethod
    def get_test_params(cls, parameter_set: str = "default"):
        """Provide deterministic toy configurations for unit tests."""
        from hyperactive.experiment.integrations import SklearnCvExperiment
        from hyperactive.experiment.toy import Ackley

        # 1) ML example (Iris + SVC)
        sklearn_exp = SklearnCvExperiment.create_test_instance()
        param_dist_1 = {
            "C": [0.01, 0.1, 1, 10],
            "gamma": np.logspace(-4, 1, 6),
        }
        params_sklearn = {
            "experiment": sklearn_exp,
            "param_distributions": param_dist_1,
            "n_iter": 5,
            "random_state": 42,
        }

        # 2) continuous optimisation example (Ackley)
        ackley_exp = Ackley.create_test_instance()
        param_dist_2 = {
            "x0": np.linspace(-5, 5, 50),
            "x1": np.linspace(-5, 5, 50),
        }
        params_ackley = {
            "experiment": ackley_exp,
            "param_distributions": param_dist_2,
            "n_iter": 20,
            "random_state": 0,
        }

        return [params_sklearn, params_ackley]
