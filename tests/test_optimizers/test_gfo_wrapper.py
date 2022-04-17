import pytest
import numpy as np

from ._parametrize import optimizers
from hyperactive.search_space import SearchSpace


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_m5(opt):
    score = -(opt["x1"] - 5) * (opt["x1"] - 5)
    return score


def objective_function_p5(opt):
    score = -(opt["x1"] + 5) * (opt["x1"] + 5)
    return score


search_space_0 = {"x1": list(np.arange(-100, 101, 1))}
search_space_1 = {"x1": list(np.arange(0, 101, 1))}
search_space_2 = {"x1": list(np.arange(-100, 1, 1))}

search_space_3 = {"x1": list(np.arange(-10, 11, 0.1))}
search_space_4 = {"x1": list(np.arange(0, 11, 0.1))}
search_space_5 = {"x1": list(np.arange(-10, 1, 0.1))}

search_space_6 = {"x1": list(np.arange(-0.0000000003, 0.0000000003, 0.0000000001))}
search_space_7 = {"x1": list(np.arange(0, 0.0000000003, 0.0000000001))}
search_space_8 = {"x1": list(np.arange(-0.0000000003, 0, 0.0000000001))}

objective_para = (
    "objective",
    [
        (objective_function),
        (objective_function_m5),
        (objective_function_p5),
    ],
)

search_space_para = (
    "search_space",
    [
        (search_space_0),
        (search_space_1),
        (search_space_2),
        (search_space_3),
        (search_space_4),
        (search_space_5),
        (search_space_6),
        (search_space_7),
        (search_space_8),
    ],
)


@pytest.mark.parametrize(*objective_para)
@pytest.mark.parametrize(*search_space_para)
@pytest.mark.parametrize(*optimizers)
def test_best_results_0(Optimizer, search_space, objective):
    search_space = search_space
    objective_function = objective

    n_iter = 10
    s_space = SearchSpace(search_space)

    initialize = {"vertices": 2}
    pass_through = {}
    callbacks = None
    catch = None
    max_score = None
    early_stopping = None
    random_state = None
    memory = None
    memory_warm_start = None
    verbosity = ["progress_bar", "print_results", "print_times"]

    opt = Optimizer()

    opt.setup_search(
        objective_function=objective_function,
        s_space=s_space,
        n_iter=n_iter,
        initialize=initialize,
        pass_through=pass_through,
        callbacks=callbacks,
        catch=catch,
        max_score=max_score,
        early_stopping=early_stopping,
        random_state=random_state,
        memory=memory,
        memory_warm_start=memory_warm_start,
        verbosity=verbosity,
    )
    opt.max_time = None
    opt.search(nth_process=0)

    assert opt.best_score == objective_function(opt.best_para)
