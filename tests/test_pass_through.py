import copy
import pytest
import numpy as np
import pandas as pd

from hyperactive import Hyperactive


search_space = {
    "x1": list(np.arange(0, 100, 1)),
}


def _test_func():
    pass


def _test_func_1():
    pass


def objective_function_0(opt):
    if opt.pass_through["stuff"] != 1:
        print("\n pass_through:", opt.pass_through["stuff"])
        assert False

    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_1(opt):
    if opt.pass_through["stuff"] != 0.001:
        print("\n pass_through:", opt.pass_through["stuff"])
        assert False

    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_2(opt):
    if opt.pass_through["stuff"] != [1, 2, 3]:
        print("\n pass_through:", opt.pass_through["stuff"])
        assert False

    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_3(opt):
    if opt.pass_through["stuff"] != _test_func:
        print("\n pass_through:", opt.pass_through["stuff"])
        assert False

    score = -opt["x1"] * opt["x1"]
    return score


pass_through_0 = {"stuff": 1}
pass_through_1 = {"stuff": 0.001}
pass_through_2 = {"stuff": [1, 2, 3]}
pass_through_3 = {"stuff": _test_func}


pass_through_setup_0 = (objective_function_0, pass_through_0)
pass_through_setup_1 = (objective_function_1, pass_through_1)
pass_through_setup_2 = (objective_function_2, pass_through_2)
pass_through_setup_3 = (objective_function_3, pass_through_3)

pass_through_setups = (
    "pass_through_setup",
    [
        (pass_through_setup_0),
        (pass_through_setup_1),
        (pass_through_setup_2),
        (pass_through_setup_3),
    ],
)


@pytest.mark.parametrize(*pass_through_setups)
def test_pass_through_0(pass_through_setup):
    objective_function = pass_through_setup[0]
    pass_through = pass_through_setup[1]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        pass_through=pass_through,
    )
    hyper.run()


def objective_function_0(opt):
    if opt.nth_iter > 1:
        assert opt.pass_through["stuff"] == 2
    opt.pass_through["stuff"] = 2

    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_1(opt):
    if opt.nth_iter > 1:
        assert opt.pass_through["stuff"] == 0.002
    opt.pass_through["stuff"] = 0.002

    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_2(opt):
    if opt.nth_iter > 1:
        assert 4 in opt.pass_through["stuff"]
    opt.pass_through["stuff"].append(4)

    score = -opt["x1"] * opt["x1"]
    return score


def objective_function_3(opt):
    if opt.nth_iter > 1:
        assert opt.pass_through["stuff"] == _test_func_1
    opt.pass_through["stuff"] = _test_func_1

    score = -opt["x1"] * opt["x1"]
    return score


pass_through_setup_0 = (objective_function_0, pass_through_0)
pass_through_setup_1 = (objective_function_1, pass_through_1)
pass_through_setup_2 = (objective_function_2, pass_through_2)
pass_through_setup_3 = (objective_function_3, pass_through_3)

pass_through_setups = (
    "pass_through_setup",
    [
        (pass_through_setup_0),
        (pass_through_setup_1),
        (pass_through_setup_2),
        (pass_through_setup_3),
    ],
)


@pytest.mark.parametrize(*pass_through_setups)
def test_pass_through_1(pass_through_setup):
    objective_function = pass_through_setup[0]
    pass_through = pass_through_setup[1]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        pass_through=pass_through,
    )
    pass_through = copy.deepcopy(pass_through)

    hyper.run()


@pytest.mark.parametrize(*pass_through_setups)
def test_pass_through_2(pass_through_setup):
    objective_function = pass_through_setup[0]
    pass_through = pass_through_setup[1]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        n_jobs=2,
        pass_through=pass_through,
    )
    pass_through = copy.deepcopy(pass_through)

    hyper.run()


@pytest.mark.parametrize(*pass_through_setups)
def test_pass_through_3(pass_through_setup):
    objective_function = pass_through_setup[0]
    pass_through = pass_through_setup[1]

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        n_jobs=4,
        pass_through=pass_through,
    )
    pass_through = copy.deepcopy(pass_through)

    hyper.run()
