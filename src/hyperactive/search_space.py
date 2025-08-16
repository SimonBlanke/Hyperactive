"""Search space utilities for hyperparameter optimization.

Author: Simon Blanke
Email: simon.blanke@yahoo.com
License: MIT License
"""

import numpy as np


class DictClass:
    """DictClass class."""

    def __init__(self, search_space):
        self.search_space = search_space

    def __getitem__(self, key):
        """Get item from search space."""
        return self.search_space[key]

    def keys(self):
        """Keys function."""
        return self.search_space.keys()

    def values(self):
        """Values function."""
        return self.search_space.values()

    def items(self):
        """Items function."""
        return self.search_space.items()


class SearchSpace(DictClass):
    """SearchSpace class."""

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
        """Return search space dictionary."""
        return self.search_space

    def dim_types(self):
        """Dim Types function."""
        data_types = {}
        for dim_key in self.dim_keys:
            dim_values = np.array(list(self.search_space[dim_key]))
            try:
                np.subtract(dim_values, dim_values)
                np.array(dim_values).searchsorted(dim_values)
            except (TypeError, ValueError):
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
                    except AttributeError:
                        func_name = value

                    func2str[dim_key].append(func_name)
        return func2str

    def check_list(self):
        """Check List function."""
        for dim_key in self.dim_keys:
            search_dim = self.search_space[dim_key]

            err_msg = (
                f"\n Value in '{dim_key}' of search space dictionary must be of "
                "type list \n"
            )
            if not isinstance(search_dim, list):
                raise ValueError(err_msg)

    @staticmethod
    def is_function(value):
        """Is Function function."""
        try:
            value.__name__
        except AttributeError:
            return False
        else:
            return True

    @staticmethod
    def is_number(value):
        """Is Number function."""
        try:
            float(value)
            value * 0.1
            value - 0.1
            value / 0.1
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        else:
            return True

    def _string_or_object(self, dim_key, dim_values):
        for dim_value in dim_values:
            is_str = isinstance(dim_value, str)
            is_func = self.is_function(dim_value)
            is_number = self.is_number(dim_value)

            if not is_str and not is_func and not is_number:
                msg = (
                    f"\n The value '{dim_value}' of type '{type(dim_value)}' in the "
                    f"search space dimension '{dim_key}' must be number, string or "
                    "function \n"
                )
                raise ValueError(msg)

    def check_non_num_values(self):
        """Check Non Num Values function."""
        for dim_key in self.dim_keys:
            dim_values = np.array(list(self.search_space[dim_key]))

            try:
                np.subtract(dim_values, dim_values)
                np.array(dim_values).searchsorted(dim_values)
            except (TypeError, ValueError):
                self._string_or_object(dim_key, dim_values)
            else:
                if dim_values.ndim != 1:
                    msg = f"Array-like object in '{dim_key}' must be one dimensional"
                    raise ValueError(msg)
