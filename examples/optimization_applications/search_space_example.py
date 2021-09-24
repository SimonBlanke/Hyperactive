"""
Hyperactive is very versatile, because it can handle not just numerical or 
string variables in the search space, but also functions. This enables many 
possibilities for more complex optimization applications. Neural architecture search,
feature engineering, ensemble optimization and many other applications are 
only possible or much easier if you can put functions in the search space.
"""

from hyperactive import Hyperactive


def function_0():
    pass


def function_1():
    pass


def function_2():
    pass


def list1():
    return [1, 0, 0]


def list2():
    return [0, 1, 0]


def list3():
    return [0, 0, 1]


# Hyperactive can handle python objects in the search space
search_space = {
    "int": list(range(1, 10)),
    "float": [0.1, 0.01, 0.001],
    "string": ["string1", "string2"],
    "function": [function_0, function_1, function_2],
    "list": [list1, list2, list3],
}


def objective_function(para):
    # score must be a number
    score = 1
    return score


hyper = Hyperactive()
hyper.add_search(objective_function, search_space, n_iter=20)
hyper.run()

search_data = hyper.results(objective_function)

print("\n Search Data: \n", search_data)
