import numpy as np
from hyperactive import Hyperactive
from hyperactive.data_tools import DataCollector


def test_data_collector_0():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=20)
    hyper.run()

    search_data = hyper.results(objective_function)

    data_c = DataCollector("./search_data.csv")
    data_c.save(search_data)
    search_data_ = data_c.load()

    search_data.equals(search_data_)


def test_data_collector_1():
    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": np.arange(-100, 101, 1),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=20)
    hyper.run()

    search_data = hyper.results(objective_function)

    data_c = DataCollector("./search_data.csv")
    data_c.save(search_data, replace_existing=True)
    search_data_ = data_c.load()

    search_data.equals(search_data_)


def test_data_collector_2():
    data_c = DataCollector("./search_data.csv")

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
    hyper.add_search(objective_function, search_space, n_iter=20)
    hyper.run()

    search_data_ = data_c.load()
