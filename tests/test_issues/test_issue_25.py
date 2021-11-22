import numpy as np
import pandas as pd

from hyperactive import Hyperactive


def test_issue_25():
    # set a path to save the dataframe
    path = "./search_data.csv"
    search_space = {
        "n_neighbors": list(range(1, 50)),
    }

    # get para names from search space + the score
    para_names = list(search_space.keys()) + ["score"]

    # init empty pandas dataframe
    search_data = pd.DataFrame(columns=para_names)
    search_data.to_csv(path, index=False)

    def objective_function(para):
        # score = random.choice([1.2, 2.3, np.nan])
        score = np.nan

        # you can access the entire dictionary from "para"
        parameter_dict = para.para_dict

        # save the score in the copy of the dictionary
        parameter_dict["score"] = score

        # append parameter dictionary to pandas dataframe
        search_data = pd.read_csv(path, na_values="nan")
        search_data_new = pd.DataFrame(parameter_dict, columns=para_names, index=[0])
        search_data = search_data.append(search_data_new)
        search_data.to_csv(path, index=False, na_rep="nan")

        return score

    hyper0 = Hyperactive()
    hyper0.add_search(objective_function, search_space, n_iter=50)
    hyper0.run()

    search_data_0 = pd.read_csv(path, na_values="nan")
    """
    the second run should be much faster than before, 
    because Hyperactive already knows most parameters/scores
    """
    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function, search_space, n_iter=50, memory_warm_start=search_data_0
    )
    hyper1.run()
