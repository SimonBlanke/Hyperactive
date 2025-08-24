"""Integration utilities for sklearn metrics with Hyperactive."""

__all__ = ["_coerce_to_scorer", "_guess_sign_of_sklmetric"]


def _coerce_to_scorer(scoring, estimator):
    """Coerce scoring argument into a sklearn scorer.

    Parameters
    ----------
    scoring : str, callable, or None
        The scoring strategy to use.
    estimator : estimator object
        The estimator to use for default scoring if scoring is None.

    Returns
    -------
    scorer : callable
        A sklearn scorer callable.
        Follows the unified sklearn scorer interface:
    """
    from sklearn.metrics import check_scoring

    # check if scoring is a scorer by checking for "estimator" in signature
    if scoring is None:
        if isinstance(estimator, str):
            if estimator == "classifier":
                from sklearn.metrics import accuracy_score

                scoring = accuracy_score
            elif estimator == "regressor":
                from sklearn.metrics import r2_score

                scoring = r2_score
        else:
            return check_scoring(estimator)

    # check using inspect.signature for "estimator" in signature
    if callable(scoring):
        from inspect import signature

        if "estimator" in signature(scoring).parameters:
            return scoring
        else:
            from sklearn.metrics import make_scorer

            return make_scorer(scoring)
    else:
        # scoring is a string (scorer name)
        return check_scoring(estimator, scoring=scoring)


def _guess_sign_of_sklmetric(scorer):
    """Guess the sign of a sklearn metric scorer.

    Parameters
    ----------
    scorer : callable
        The sklearn metric scorer to guess the sign for.

    Returns
    -------
    int
        1 if higher scores are better, -1 if lower scores are better.
    """
    HIGHER_IS_BETTER = {
        # Classification
        "accuracy_score": True,
        "auc": True,
        "average_precision_score": True,
        "balanced_accuracy_score": True,
        "brier_score_loss": False,
        "class_likelihood_ratios": False,
        "cohen_kappa_score": True,
        "d2_log_loss_score": True,
        "dcg_score": True,
        "f1_score": True,
        "fbeta_score": True,
        "hamming_loss": False,
        "hinge_loss": False,
        "jaccard_score": True,
        "log_loss": False,
        "matthews_corrcoef": True,
        "ndcg_score": True,
        "precision_score": True,
        "recall_score": True,
        "roc_auc_score": True,
        "top_k_accuracy_score": True,
        "zero_one_loss": False,
        # Regression
        "d2_absolute_error_score": True,
        "d2_pinball_score": True,
        "d2_tweedie_score": True,
        "explained_variance_score": True,
        "max_error": False,
        "mean_absolute_error": False,
        "mean_absolute_percentage_error": False,
        "mean_gamma_deviance": False,
        "mean_pinball_loss": False,
        "mean_poisson_deviance": False,
        "mean_squared_error": False,
        "mean_squared_log_error": False,
        "mean_tweedie_deviance": False,
        "median_absolute_error": False,
        "r2_score": True,
        "root_mean_squared_error": False,
        "root_mean_squared_log_error": False,
    }

    scorer_name = getattr(scorer, "__name__", None)

    if hasattr(scorer, "greater_is_better"):
        return 1 if scorer.greater_is_better else -1
    elif scorer_name in HIGHER_IS_BETTER:
        return 1 if HIGHER_IS_BETTER[scorer_name] else -1
    elif scorer_name.endswith("_score"):
        # If the scorer name ends with "_score", we assume higher is better
        return 1
    elif scorer_name.endswith("_loss") or scorer_name.endswith("_deviance"):
        # If the scorer name ends with "_loss", we assume lower is better
        return -1
    elif scorer_name.endswith("_error"):
        return -1
    else:
        # If we cannot determine the sign, we assume lower is better
        return -1
