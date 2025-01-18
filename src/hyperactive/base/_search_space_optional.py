import numpy as np
from hyperactive import BaseSearchSpace


"""
Optional:

It might make sense to use a class instead of a dictionary for the search-space.
The search-space can have specifc properties, that can be computed from the params (previously called search-space-dictionary).
The search-space can have a certain size, has n dimensions, some of which are numeric, some of which are categorical.
"""


class SearchSpace:
    search_space: dict = None

    def __init__(self, **params):
        super().__init__()

        for key, value in params.items():
            setattr(self, key, value)
            self.search_space[key] = value

    def __call__(self):
        return self.search_space
