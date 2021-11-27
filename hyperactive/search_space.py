# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class DictClass:
    def __init__(self, search_space):
        self.search_space = search_space

    def __getitem__(self, key):
        return self.search_space[key]

    def keys(self):
        return self.search_space.keys()

    def values(self):
        return self.search_space.values()


class SearchSpace(DictClass):
    def __init__(self, search_space):
        super().__init__(search_space)
        self.search_space = search_space

        self.dim_keys = list(search_space.keys())
        self.values_l = list(self.search_space.values())

        positions = {}
        for key in search_space.keys():
            positions[key] = np.array(range(len(search_space[key])))
        self.positions = positions

        self.check_list()
        self.check_non_num_values()

        self.data_types = self.dim_types()
        self.func2str = self._create_num_str_ss()

    def __call__(self):
        return self.search_space

    def dim_types(self):
        data_types = {}
        for dim_key in self.dim_keys:
            dim_values = np.array(list(self.search_space[dim_key]))
            try:
                np.subtract(dim_values, dim_values)
                np.array(dim_values).searchsorted(dim_values)
            except:
                _type_ = "object"
            else:
                _type_ = "number"

            data_types[dim_key] = _type_
        return data_types

    def _create_num_str_ss(self):
        func2str = {}
        for dim_key in self.dim_keys:
            if self.data_types[dim_key] == "number":
                func2str[dim_key] = self.search_space[dim_key]
            else:
                func2str[dim_key] = []

                dim_values = self.search_space[dim_key]
                for value in dim_values:
                    try:
                        func_name = value.__name__
                    except:
                        func_name = value

                    func2str[dim_key].append(func_name)
        return func2str

    def check_list(self):
        for dim_key in self.dim_keys:
            search_dim = self.search_space[dim_key]

            err_msg = "\n Value in '{}' of search space dictionary must be of type list \n".format(
                dim_key
            )
            if not isinstance(search_dim, list):
                print("Warning: ", err_msg)
                # raise ValueError(err_msg)

    @staticmethod
    def is_function(value):
        try:
            value.__name__
        except:
            return False
        else:
            return True

    def _string_or_object(self, dim_key, dim_values):
        for dim_value in dim_values:
            is_str = isinstance(dim_value, str)
            is_func = self.is_function(dim_value)

            if not is_str and not is_func:
                msg = "\n The value '{}' of type '{}' in the search space dimension '{}' must be number, string or function \n".format(
                    dim_value, type(dim_value), dim_key
                )
                print("Warning: ", msg)
                # raise ValueError(msg)

    def check_non_num_values(self):
        for dim_key in self.dim_keys:
            dim_values = np.array(list(self.search_space[dim_key]))

            try:
                np.subtract(dim_values, dim_values)
                np.array(dim_values).searchsorted(dim_values)
            except:
                self._string_or_object(dim_key, dim_values)
            else:
                if dim_values.ndim != 1:
                    msg = "Array-like object in '{}' must be one dimensional".format(
                        dim_key
                    )
                    print("Warning: ", msg)
                    # raise ValueError(msg)
