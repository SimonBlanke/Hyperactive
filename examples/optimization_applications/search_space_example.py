import numpy as np
import pandas as pd
from hyperactive import Hyperactive


def function_():
    pass


class class_:
    def __init__(self):
        pass


search_space = {
    "int": list(range(1, 10)),
    "float": [0.1, 0.01, 0.001],
    "string": ["string1", "string2"],
    "function": [function_],
    "class": [class_],
    "list": [[1, 1, 1], [1, 1, 2], [1, 2, 1]],
    "numpy": [np.array([1, 2, 3])],
    "pandas": [pd.DataFrame([[1, 2], [3, 4]], columns=["y1", "y2"])],
}


def objective_function(para):
    # score must be a single number
    score = 1
    return score


hyper = Hyperactive()
hyper.add_search(objective_function, search_space, n_iter=20)
hyper.run()

search_data = hyper.results(objective_function)

for col_name in search_data.columns:
    print("\nColumn name:", col_name, "\n", search_data[col_name][0])
