import os
import numpy as np
import pandas as pd
from hyperactive import Hyperactive
from hyperactive.dashboards import ProgressBoard
from hyperactive.dashboards.progress_board.streamlit_backend import StreamlitBackend
from hyperactive.dashboards.progress_board.progress_io import ProgressIO


def test_progress_io_0():
    search_id = "test_model"

    _io_ = ProgressIO("./")
    _io_.get_filter_file_path(search_id)


def test_progress_io_1():
    search_id = "test_model"

    _io_ = ProgressIO("./")
    _io_.get_progress_data_path(search_id)


def test_progress_io_2():
    search_id = "test_model"

    _io_ = ProgressIO("./")
    _io_.remove_filter(search_id)
    filter_ = _io_.load_filter(search_id)

    assert filter_ is None


def test_progress_io_3():
    search_id = "test_model"

    _io_ = ProgressIO("./")
    _io_.remove_progress(search_id)
    progress_ = _io_.load_progress(search_id)

    assert progress_ is None


def test_progress_io_4():
    search_id = "test_model"

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    _io_ = ProgressIO("./")
    _io_.remove_progress(search_id)
    _io_.create_filter(search_id, search_space)

    progress_ = _io_.load_filter(search_id)

    assert progress_ is not None


def test_progress_board_0():
    search_id = "test_model"

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    board = ProgressBoard()
    board.init_paths(search_id=search_id, search_space=search_space)


def test_streamlit_backend_0():
    search_id1 = "test_model1"
    search_id2 = "test_model2"
    search_id3 = "test_model3"

    search_ids = [search_id1, search_id2, search_id3]

    board = StreamlitBackend(search_ids)
    progress_data = board.get_progress_data(search_id1)

    assert progress_data is None


def test_filter_data_0():
    search_id1 = "test_model1"
    search_id2 = "test_model2"
    search_id3 = "test_model3"
    search_ids = [search_id1, search_id2, search_id3]

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=200)
    hyper.run()
    search_data = hyper.results(objective_function)

    indices = list(search_space.keys()) + ["score"]
    filter_dict = {
        "parameter": indices,
        "lower bound": "lower",
        "upper bound": "upper",
    }
    filter_df = pd.DataFrame(filter_dict)
    threshold = -1000
    filter_df.iloc[1, 1] = threshold

    board = StreamlitBackend(search_ids)
    progress_data = board.filter_data(search_data, filter_df)

    assert not np.all(search_data["score"].values >= threshold)
    assert np.all(progress_data["score"].values >= threshold)
