"""constraint module for Hyperactive optimization."""

# Email: simon.blanke@yahoo.com
# License: MIT License


def gfo2hyper(search_space, para):
    """Gfo2Hyper function."""
    values_dict = {}
    for key, values in search_space.items():
        pos_ = int(para[key])
        values_dict[key] = values[pos_]

    return values_dict


class Constraint:
    """Constraint class."""

    def __init__(self, constraint, search_space):
        self.constraint = constraint
        self.search_space = search_space

    def __call__(self, para):
        """Call constraint on parameters."""
        para = gfo2hyper(self.search_space, para)
        return self.constraint(para)
