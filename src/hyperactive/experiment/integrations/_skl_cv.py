"""Integration utilities for sklearn splitters with Hyperactive."""

__all__ = ["_coerce_cv"]


def _coerce_cv(cv):
    """Coerce cv argument into a sklearn-compatible cv splitter.

    Parameters
    ----------
    cv : int, cross-validation generator, or iterable
        The cross-validation strategy to use.

    Returns
    -------
    cv_splitter : cross-validation generator or iterable
        A sklearn-compatible cross-validation splitter.
    """
    from sklearn.model_selection import KFold

    # default handling for cv
    if isinstance(cv, int):
        from sklearn.model_selection import KFold

        return KFold(n_splits=cv, shuffle=True)
    elif cv is None:
        from sklearn.model_selection import KFold

        return KFold(n_splits=3, shuffle=True)
    else:
        return cv
