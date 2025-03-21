import time
import pytest
import numpy as np
import pandas as pd

from hyperactive import Hyperactive

from ._search_space_list import search_space_setup

search_space_list = search_space_setup()


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


@pytest.mark.parametrize("search_space", search_space_list)
def test_warm_start_0(search_space):
    hyper0 = Hyperactive()
    hyper0.add_search(objective_function, search_space, n_iter=20)
    hyper0.run()

    search_data0 = hyper0.best_para(objective_function)

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function,
        search_space,
        n_iter=20,
        initialize={"warm_start": [search_data0]},
    )
    hyper1.run()



