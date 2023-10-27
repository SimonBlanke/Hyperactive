import numpy as np

from hyperactive import Hyperactive


def convex_function(pos_new):
    score = -(pos_new["x1"] * pos_new["x1"] + pos_new["x2"] * pos_new["x2"])
    return score


search_space = {
    "x1": list(np.arange(-100, 101, 0.1)),
    "x2": list(np.arange(-100, 101, 0.1)),
}


def constraint_1(para):
    # reject parameters where x1 and x2 are higher than 2.5 at the same time
    return not (para["x1"] > 2.5 and para["x2"] > 2.5)


# put one or more constraints inside a list
constraints_list = [constraint_1]


hyper = Hyperactive()
# pass list of constraints
hyper.add_search(
    convex_function,
    search_space,
    n_iter=50,
    constraints=constraints_list,
)
hyper.run()

search_data = hyper.search_data(convex_function)

print("\n search_data \n", search_data, "\n")
