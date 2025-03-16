import numpy as np
import pandas as pd


def search_space_setup(size=1000):
    if size < 1000:
        print("Error: Some search spaces cannot be created")
        return

    pad_full = list(range(0, size))
    pad_cat = list(range(int(size / 3)))
    pad_10 = list(range(int(size ** 0.1)))

    search_space_0 = {
        "x1": pad_full,
    }

    search_space_1 = {
        "x1": pad_cat,
        "x2": list(range(3)),
    }

    search_space_2 = {
        "x1": pad_10,
        "x2": pad_10,
        "x3": pad_10,
        "x4": pad_10,
        "x5": pad_10,
        "x6": pad_10,
        "x7": pad_10,
        "x8": pad_10,
        "x9": pad_10,
        "x10": pad_10,
    }

    search_space_3 = {
        "x1": pad_10,
        "x2": pad_10,
        "x3": pad_10,
        "x4": pad_10,
        "x5": pad_10,
        "x6": pad_10,
        "x7": pad_10,
        "x8": pad_10,
        "x9": pad_10,
        "x10": pad_10,
        "x11": [1],
        "x12": [1],
        "x13": [1],
        "x14": [1],
        "x15": [1],
        "x16": [1],
        "x17": [1],
        "x18": [1],
        "x19": [1],
        "x20": [1],
    }

    search_space_4 = {
        "x1": pad_cat,
        "str1": ["0", "1", "2"],
    }

    def func1():
        pass

    def func2():
        pass

    def func3():
        pass

    search_space_5 = {
        "x1": pad_cat,
        "func1": [func1, func2, func3],
    }

    class class1:
        pass

    class class2:
        pass

    class class3:
        pass

    def wr_func_1():
        return class1

    def wr_func_2():
        return class2

    def wr_func_3():
        return class3

    search_space_6 = {
        "x1": pad_cat,
        "class_1": [wr_func_1, wr_func_2, wr_func_3],
    }

    class class1:
        def __init__(self):
            pass

    class class2:
        def __init__(self):
            pass

    class class3:
        def __init__(self):
            pass

    def wr_func_1():
        return class1()

    def wr_func_2():
        return class2()

    def wr_func_3():
        return class3()

    search_space_7 = {
        "x1": pad_cat,
        "class_obj_1": [wr_func_1, wr_func_2, wr_func_3],
    }

    def wr_func_1():
        return [1, 0, 0]

    def wr_func_2():
        return [0, 1, 0]

    def wr_func_3():
        return [0, 0, 1]

    search_space_8 = {
        "x1": pad_cat,
        "list_1": [wr_func_1, wr_func_2, wr_func_3],
    }

    def wr_func_1():
        return np.array([1, 0, 0])

    def wr_func_2():
        return np.array([0, 1, 0])

    def wr_func_3():
        return np.array([0, 0, 1])

    search_space_9 = {
        "x1": pad_cat,
        "array_1": [wr_func_1, wr_func_2, wr_func_3],
    }

    def wr_func_1():
        return pd.DataFrame(np.array([1, 0, 0]))

    def wr_func_2():
        return pd.DataFrame(np.array([0, 1, 0]))

    def wr_func_3():
        return pd.DataFrame(np.array([0, 0, 1]))

    search_space_10 = {
        "x1": pad_cat,
        "df_1": [wr_func_1, wr_func_2, wr_func_3],
    }

    search_space_list = [
        (search_space_0),
        (search_space_1),
        (search_space_2),
        (search_space_3),
        (search_space_4),
        (search_space_5),
        (search_space_6),
        (search_space_7),
        (search_space_8),
        (search_space_9),
        (search_space_10),
    ]

    return search_space_list
