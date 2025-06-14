# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def gfo2hyper(search_space, para):
    values_dict = {}
    for key, values in search_space.items():
        pos_ = int(para[key])
        values_dict[key] = values[pos_]

    return values_dict


class Constraint:
    def __init__(self, constraint, search_space):
        self.constraint = constraint
        self.search_space = search_space

    def __call__(self, para):
        para = gfo2hyper(self.search_space, para)
        return self.constraint(para)
