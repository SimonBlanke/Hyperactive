import os
import inspect
import pytest
import numpy as np
import pandas as pd

from hyperactive import Hyperactive, LongTermMemory


def remove_file(model_name):
    try:
        os.remove("./" + model_name + ".pkl")
    except OSError:
        pass


def func1():
    pass


def func2():
    pass


def func3():
    pass


def objective_function(opt):
    score = -opt["x1"] * opt["x1"]
    return score


class class1:
    pass


class class2:
    pass


class class3:
    pass


class class1_:
    def __init__(self):
        pass


class class2_:
    def __init__(self):
        pass


class class3_:
    def __init__(self):
        pass


search_space_int = {
    "x1": list(range(0, 3, 1)),
}

search_space_float = {
    "x1": list(np.arange(0, 0.003, 0.001)),
}

search_space_str = {
    "x1": list(range(0, 100, 1)),
    "str1": ["0", "1", "2"],
}

search_space_func = {
    "x1": list(range(0, 100, 1)),
    "func1": [func1, func2, func3],
}


search_space_class = {
    "x1": list(range(0, 100, 1)),
    "class1": [class1, class2, class3],
}


search_space_obj = {
    "x1": list(range(0, 100, 1)),
    "class1": [class1_(), class2_(), class3_()],
}

search_space_lists = {
    "x1": list(range(0, 100, 1)),
    "list1": [[1, 1, 1], [1, 2, 1], [1, 1, 2]],
}

search_space_para = (
    "search_space",
    [
        (search_space_int),
        (search_space_float),
        (search_space_str),
        (search_space_func),
        (search_space_class),
        (search_space_obj),
        (search_space_lists),
    ],
)

path_para = (
    "path",
    [("./"), (None),],
)


def test_ltm_0():
    model_name = "test_ltm_0"

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    search_space = {
        "x1": list(range(0, 3, 1)),
    }

    hyper = Hyperactive()
    hyper.add_search(objective_function, search_space, n_iter=25)
    hyper.run()

    memory = LongTermMemory(model_name, path="./")
    results1 = hyper.results(objective_function)
    memory.save(results1)

    results2 = memory.load()
    remove_file(model_name)

    assert results1.equals(results2)


@pytest.mark.parametrize(*path_para)
@pytest.mark.parametrize(*search_space_para)
def test_ltm_1(search_space, path):
    model_name = "test_ltm_1"

    def objective_function(opt):
        score = -opt["x1"] * opt["x1"]
        return score

    memory = LongTermMemory(model_name, path=path)

    hyper0 = Hyperactive()
    hyper0.add_search(
        objective_function, search_space, n_iter=25, long_term_memory=memory
    )
    hyper0.run()

    hyper1 = Hyperactive()
    hyper1.add_search(
        objective_function, search_space, n_iter=25, long_term_memory=memory
    )
    hyper1.run()

    remove_file(model_name)


def test_ltm_int():
    model_name = "test_ltm_int"
    array = np.arange(3 * 10).reshape(10, 3)
    results1 = pd.DataFrame(array, columns=["x1", "x2", "x3"])

    memory = LongTermMemory(model_name, path="./")
    memory.save(results1)

    results2 = memory.load()

    remove_file(model_name)

    assert results1.equals(results2)


def test_ltm_float():
    model_name = "test_ltm_float"
    array = np.arange(3 * 10).reshape(10, 3)
    array = array / 1000
    results1 = pd.DataFrame(array, columns=["x1", "x2", "x3"])

    memory = LongTermMemory(model_name, path="./")
    memory.save(results1)

    results2 = memory.load()

    remove_file(model_name)

    assert results1.equals(results2)


def test_ltm_str():
    model_name = "test_ltm_str"
    array = ["str1", "str2", "str3"]
    results1 = pd.DataFrame(
        [array, array, array, array, array], columns=["x1", "x2", "x3"]
    )

    memory = LongTermMemory(model_name, path="./")
    memory.save(results1)

    results2 = memory.load()

    remove_file(model_name)

    assert results1.equals(results2)


def func1():
    pass


def func2():
    pass


def func3():
    pass


def test_ltm_func():
    model_name = "test_ltm_func"

    array = [func1, func2, func3]
    results1 = pd.DataFrame(
        [array, array, array, array, array], columns=["x1", "x2", "x3"]
    )

    memory = LongTermMemory(model_name, path="./")
    memory.save(results1)

    results2 = memory.load()

    remove_file(model_name)

    func_str_list = []
    for func_ in array:
        func_str_list.append(inspect.getsource(func_))

    for func_ in list(results2.values.flatten()):
        func_str_ = inspect.getsource(func_)
        if func_str_ not in func_str_list:
            assert False


class class1:
    name = "class1"


class class2:
    name = "class2"


class class3:
    name = "class3"


def test_ltm_class():
    model_name = "test_ltm_class"

    array = [class1, class2, class3]
    results1 = pd.DataFrame(
        [array, array, array, array, array], columns=["x1", "x2", "x3"]
    )

    memory = LongTermMemory(model_name, path="./")
    memory.save(results1)

    results2 = memory.load()

    remove_file(model_name)

    for class_1 in list(results2.values.flatten()):
        assert_ = False
        for class_2 in array:
            if class_1.name == class_2.name:
                assert_ = True
                break

        assert assert_


class class1_:
    def __init__(self):
        self.name = "class1_"


class class2_:
    def __init__(self):
        self.name = "class2_"


class class3_:
    def __init__(self):
        self.name = "class3_"


def test_ltm_obj():
    model_name = "test_ltm_obj"

    array = [class1_(), class2_(), class3_()]
    results1 = pd.DataFrame(
        [array, array, array, array, array], columns=["x1", "x2", "x3"]
    )

    memory = LongTermMemory(model_name, path="./")
    memory.save(results1)

    results2 = memory.load()

    remove_file(model_name)

    for obj_1 in list(results2.values.flatten()):
        assert_ = False
        for obj_2 in array:
            if obj_1.name == obj_2.name:
                assert_ = True
                break

        assert assert_


def test_ltm_list():
    model_name = "test_ltm_list"

    dict_ = {"x1": [[1, 1, 1], [1, 2, 1], [1, 1, 2]]}
    results1 = pd.DataFrame(dict_)

    memory = LongTermMemory(model_name, path="./")
    memory.save(results1)

    results2 = memory.load()

    remove_file(model_name)

    for list_1 in list(results2.values.flatten()):
        assert_ = False
        for list_2 in list(dict_.values())[0]:
            if list_1 == list_2:
                assert_ = True
                break

        assert assert_
