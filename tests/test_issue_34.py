import numpy as np
from hyperactive import Hyperactive


def test_mixed_type_search_space():
    def objective_function(para):
        assert isinstance(para["x1"], int)
        assert isinstance(para["x2"], float)

        return 1

    search_space = {
        "x1": list(range(10, 20)),
        "x2": list(np.arange(1, 2, 0.1)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=10000)
    hyper.run()
