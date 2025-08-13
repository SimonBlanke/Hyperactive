"""This module contains common functions used by multiple optimizers."""

__all__ = ["_score_params"]


def _score_params(params, meta):
    """Function to score parameters, used in parallelization."""
    meta = meta.copy()
    experiment = meta["experiment"]
    error_score = meta["error_score"]

    try:
        return experiment(**params)
    except Exception:  # noqa: B904
        # Catch all exceptions and assign error_score
        return error_score
