import numpy as np
from hyperactive import Hyperactive
from hyperactive.dashboards import ProgressBoard
from hyperactive.dashboards.progress_board.streamlit_backend import StreamlitBackend


from .config import n_iter, n_jobs

search_ids = ["objective_function"]
backend = StreamlitBackend(search_ids)


def objective_function(opt):
    score = -(opt["x1"] * opt["x1"])

    for search_id in search_ids:
        pyplot_fig, plotly_fig = backend.create_plots(search_id)

    return score


search_space = {
    "x1": np.arange(-100, 101, 1),
    "x2": np.arange(-100, 101, 1),
    "x3": np.arange(-100, 101, 1),
    "x4": np.arange(-100, 101, 1),
    "x5": np.arange(-100, 101, 1),
    "x6": np.arange(-100, 101, 1),
    "x7": np.arange(-100, 101, 1),
    "x8": np.arange(-100, 101, 1),
    "x9": np.arange(-100, 101, 1),
}


def test_progress_board__1():
    prog = ProgressBoard()

    hyper = Hyperactive()
    hyper.add_search(
        objective_function,
        search_space,
        n_iter=n_iter,
        n_jobs=n_jobs,
        progress_board=prog,
    )
    hyper.run(_test_st_backend=True)
