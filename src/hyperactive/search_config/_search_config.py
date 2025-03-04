"""
Optional:

It might make sense to use a class instead of a dictionary for the search-space.
The search-space can have specifc properties, that can be computed from the params (previously called search-space-dictionary).
The search-space can have a certain size, has n dimensions, some of which are numeric, some of which are categorical.
"""

from typing import Union, List, Dict, Type
from collections.abc import MutableMapping

from ._properties import calculate_properties


class SearchSpaceDictLike(MutableMapping):
    _search_space: dict = None

    def __init__(self, constraints: List[callable] = None, **param_space):
        self._search_space = dict(**param_space)
        self.constraints = constraints

    def __getitem__(self, key):
        return self._search_space[key]

    def __setitem__(self, key, value):
        self._search_space[key] = value

    def __delitem__(self, key):
        del self._search_space[key]

    def __iter__(self):
        return iter(self._search_space)

    def __len__(self):
        return len(self._search_space)

    def __repr__(self):
        return f"{self.__class__.__name__}({self._search_space})"

    def __call__(self):
        return self._search_space

    def keys(self):
        return self._search_space.keys()

    def values(self):
        return self._search_space.values()

    def items(self):
        return self._search_space.items()


class SearchConfig(SearchSpaceDictLike):

    @calculate_properties
    def __init__(self, **param_space):
        super().__init__(**param_space)

        for key, value in param_space.items():
            setattr(self, key, value)

    @calculate_properties
    def __setitem__(self, key, value):
        SearchSpaceDictLike.__setitem__(self, key, value)

    @calculate_properties
    def __delitem__(self, key):
        SearchSpaceDictLike.__delitem__(self, key)

    def print(self, indent=2, max_list_length=5):
        """
        Prints the dictionary in a readable format.

        Args:
            indent (int): The number of spaces to indent nested structures.
            max_list_length (int): The maximum number of items to display from long lists.
        """

        def format_value(value, level=0):
            prefix = " " * (level * indent)
            if isinstance(value, list):
                if len(value) > max_list_length:
                    # Truncate long lists for readability
                    result = "[\n"
                    result += "".join(
                        f"{prefix}{' ' * indent}{repr(item)},\n"
                        for item in value[:max_list_length]
                    )
                    result += f"{prefix}{' ' * indent}... ({len(value) - max_list_length} more items)\n"
                    result += f"{prefix}]"
                else:
                    result = "[\n"
                    result += "".join(
                        f"{prefix}{' ' * indent}{repr(item)},\n"
                        for item in value
                    )
                    result += f"{prefix}]"
            elif isinstance(value, dict):
                # Format nested dictionaries
                result = "{\n"
                for k, v in value.items():
                    result += f"{prefix}{' ' * indent}{repr(k)}: {format_value(v, level + 1)},\n"
                result += f"{prefix}}}"
            else:
                result = repr(value)
            return result

        for key, value in self.items():
            print(f"{key}: {format_value(value)}")
