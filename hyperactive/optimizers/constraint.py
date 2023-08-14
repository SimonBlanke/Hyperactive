# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def gfo2hyper(search_space, para):
    values_dict = {}
    for _, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class Constraint:
    def __init__(self, constraint):
        self.constraint = constraint

    def __call__(self, search_space):
        # wrapper for GFOs
        def _constr(para):
            print("\n para", para)
            para = gfo2hyper(search_space, para)
            print(" para", para)

            return self.constraint(para)

        _constr.__name__ = self.constraint.__name__
        return _constr
