"""Test that old function signature can be passed as experiment."""
# copyright: hyperactive developers, MIT License (see LICENSE file)


def test_dynamic_exp():
    """Test that old function signature can be passed as experiment."""

    # 1. define the experiment
    def parabola(opt):
        return opt["x"] ** 2 + opt["y"] ** 2

    # 2. set up the HillClimbing optimizer
    import numpy as np

    from hyperactive.opt import HillClimbing

    hillclimbing_config = {
        "search_space": {
            "x": np.array([0, 1, 2]),
            "y": np.array([0, 1, 2]),
        },
    }
    hill_climbing = HillClimbing(**hillclimbing_config, experiment=parabola)

    # 3. run the HillClimbing optimizer
    hill_climbing.run()

    best_params = hill_climbing.best_params_
    assert best_params is not None, "Best parameters should not be None"
    assert isinstance(best_params, dict), "Best parameters should be a dictionary"
    assert "x" in best_params, "Best parameters should contain 'x'"
    assert "y" in best_params, "Best parameters should contain 'y'"
