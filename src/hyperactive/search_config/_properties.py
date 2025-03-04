import numpy as np


def n_dim(search_space):
    return len(search_space)


def dim_names(search_space):
    return list(search_space.keys())


def position_space(search_space):
    return {
        key: np.array(range(len(search_space[key])))
        for key in search_space.keys()
    }


def calculate_properties(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)

        self.n_dim = n_dim(self._search_space)
        self.dim_names = dim_names(self._search_space)
        self.position_space = position_space(self._search_space)

    return wrapper
