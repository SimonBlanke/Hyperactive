"""Integration tests for skpro tuners."""
# copyright: hyperactive developers, MIT License (see LICENSE file)

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("skpro", severity="none"):
    from hyperactive.integrations.skpro import ProbaRegOptCV

    EST_TO_TEST = [ProbaRegOptCV]
else:
    EST_TO_TEST = []


@pytest.mark.parametrize("estimator", EST_TO_TEST)
def test_sktime_estimator(estimator):
    """Test sktime estimator via check_estimator."""
    from skpro.utils.estimator_checks import check_estimator

    check_estimator(estimator, raise_exceptions=True)
    # The above line collects all API conformance tests in skpro and runs them.
    # It will raise an error if the estimator is not API conformant.
