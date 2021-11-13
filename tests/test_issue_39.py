import numpy as np

from hyperactive import Hyperactive


class Parameters:
    def __init__(self):
        self.x = 5


finp = Parameters()


def ret_df():
    pass


def func_minl(opts):
    return opts["slope"] + opts["exp"]


initialize = {
    "warm_start": [
        {
            "exp": 2,
            "slope": 5,
            "freq_mult": 1.5,
            "clust": 5,
            "df": ret_df,
            "finp": finp,
            "asc": False,
            "use_pca": False,
            "last": False,
            "disc_type": "type",
        }
    ]
}


search_space = {
    "exp": list(range(0, 5)),
    "slope": list(np.arange(0.001, 10, step=0.05)),
    "freq_mult": list(np.arange(1, 2.5, 0.005)),
    "clust": [5],
    "df": [ret_df],
    "finp": [finp],
    "asc": [False],
    "use_pca": [False],
    "last": [False],
    "disc_type": ["type"],
}


h = Hyperactive(["progress_bar", "print_results", "print_times"])
h.add_search(
    func_minl,
    search_space=search_space,
    n_iter=10,
    initialize=initialize,
    memory=True,
    memory_warm_start=None,
)
