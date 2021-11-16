# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np


class SearchSpace:
    def __init__(self, search_space):
        self.search_space = search_space
        self.dim_keys = list(search_space.keys())

        self.check_list()
        self.data_types = self.dim_types()

    def __call__(self):
        return self.search_space

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

    def string_or_object(self, dim_values):
        types_l = []
        for dim_value in dim_values:
            if isinstance(dim_value, str):
                types_l.append("string")
            elif self.is_function(dim_value):
                types_l.append("function")
            else:
                err_msg = "\n The value '{}' in the search space must be number, string or function \n".format(
                    dim_value
                )
                print("Warning: ", err_msg)
                # raise ValueError(err_msg)

        return types_l

    def dim_types(self):
        data_types = {}
        for dim_key in self.dim_keys:
            dim_values = np.array(list(self.search_space[dim_key]))
            try:
                np.subtract(dim_values, dim_values)
            except:
                _type_ = self.string_or_object(dim_values)
            else:
                _type_ = "number"

            data_types[dim_key] = _type_
        return data_types
