"""Common functions used by multiple optimizers."""

__all__ = ["_score_params"]


def _score_params(params, meta):
    """Evaluate parameters (raw evaluate), used in parallelization.

    Returns the raw evaluate() value (not sign-adjusted), so that upstream
    selection can consistently use the experiment tag to choose min/max.
    """
    meta = meta.copy()
    experiment = meta["experiment"]
    error_score = meta["error_score"]

    try:
        value, _ = experiment.evaluate(params)
        return float(value)
    except Exception:  # noqa: B904
        # Catch all exceptions and assign error_score
        return error_score
