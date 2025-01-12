import numpy as np
from hyperactive import BaseSearchSpace


class SearchSpace:
    search_space: dict = None

    def __init__(self, **params):
        super().__init__()

        for key, value in params.items():
            setattr(self, key, value)
            self.search_space[key] = value

    def __call__(self):
        return self.search_space
