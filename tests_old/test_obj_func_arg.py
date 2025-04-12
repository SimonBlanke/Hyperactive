import numpy as np
from hyperactive import Hyperactive


search_space = {
    "x1": list(np.arange(0, 100, 1)),
}


def test_argument_0():
    def objective_function(para):

        print("\npara.nth_iter", para.nth_iter)
        print("nth_iter_local", para.pass_through["nth_iter_local"])

        assert para.nth_iter == para.pass_through["nth_iter_local"]

        para.pass_through["nth_iter_local"] += 1

        return 0

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=100,
        pass_through={"nth_iter_local": 0},
        memory=False,
    )
    hyper.run()
