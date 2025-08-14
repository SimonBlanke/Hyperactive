# copied from sktime, BSD-3-Clause License (see LICENSE file)
# to be moved to scikit-base in the future
"""Tests for parallelization utilities."""
import copy
import os

import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from hyperactive.utils.parallel import _get_parallel_test_fixtures, parallelize


@pytest.mark.skipif(
    not _check_soft_dependencies("ray", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
def test_ray_leaves_params_invariant():
    """Test that the parallelize function leaves backend_params invariant."""

    def trial_function(params, meta):
        return params

    backend = "ray"
    backend_params = {
        "mute_warnings": True,
        "ray_remote_args": {"num_cpus": os.cpu_count() - 1},
    }
    # copy for later comparison
    backup = backend_params.copy()

    params = [1, 2, 3]
    meta = {}

    parallelize(trial_function, params, meta, backend, backend_params)

    assert backup == backend_params


def square(x, **kwargs):
    """Square function, for testing."""
    return x**2


@pytest.mark.parametrize("fixture", _get_parallel_test_fixtures())
def test_parallelize_simple_loop(fixture):
    """Test that parallelize works with a simple function and fixture."""
    backend = fixture["backend"]
    backend_params = copy.deepcopy(fixture["backend_params"])
    params_before = copy.deepcopy(fixture["backend_params"])

    nums = range(8)
    expected = [x**2 for x in nums]

    result = parallelize(
        square,
        nums,
        backend=backend,
        backend_params=backend_params,
    )

    assert list(result) == expected
    assert backend_params == params_before
