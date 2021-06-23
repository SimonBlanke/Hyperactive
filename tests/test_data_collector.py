import os
import numpy as np
from hyperactive import Hyperactive
from hyperactive.data_tools import DataCollector


def df_equal(df1, df2):
    scores1 = df1["score"].values
    scores2 = df2["score"].values

    print("\n scores2 \n", scores2, "\n", scores1.sum())
    print("\n scores2 \n", scores2, "\n", scores2.sum())

    if abs(scores1.sum() - scores2.sum()) < 0.0000001:
        return True
    else:
        return False


def test_data_collector_0():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=50)
    hyper.run()

    search_data = hyper.results(objective_function)

    data_c = DataCollector("./search_data.csv")
    data_c.save(search_data)
    search_data_ = data_c.load()

    assert df_equal(search_data, search_data_)


def test_data_collector_1():
    path = "./search_data.csv"
    if os.path.isfile(path):
        os.remove(path)

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=50)
    hyper.run()

    search_data = hyper.results(objective_function)

    data_c = DataCollector(path)
    data_c.save(search_data, replace_existing=True)
    search_data_ = data_c.load()

    assert df_equal(search_data, search_data_)


def test_data_collector_2():
    path = "./search_data.csv"
    if os.path.isfile(path):
        os.remove(path)

    data_c = DataCollector(path, drop_duplicates=False)

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]

        para_dict = {
            "x1": opt["x1"],
            "score": score,
        }

        data_c.append(para_dict)

        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=50, memory=False)
    hyper.run()
    search_data = hyper.results(objective_function)[["x1", "score"]]

    search_data_ = data_c.load()

    assert df_equal(search_data, search_data_)


path = "./search_data.csv"
data_c = DataCollector(path, drop_duplicates=["x1"])


def objective_function_3(opt):
    score = -opt["x1"] * opt["x1"]

    para_dict = {
        "x1": opt["x1"],
        "score": score,
    }

    data_c.append(para_dict)

    return score


def test_data_collector_3():
    if os.path.isfile(path):
        os.remove(path)

    search_space = {
        "x1": np.arange(-100, 101, 0.001),
    }

    hyper = Hyperactive()
    hyper.add_search(
        objective_function_3,
        search_space,
        n_iter=5,
        n_jobs=2,
        memory=False,
        initialize={"random": 1},
    )
    hyper.run()
    search_data = hyper.results(objective_function_3)

    search_data_ = data_c.load()

    print("\n search_data \n", search_data)
    print("\n search_data_ \n", search_data_)

    assert df_equal(search_data, search_data_)
