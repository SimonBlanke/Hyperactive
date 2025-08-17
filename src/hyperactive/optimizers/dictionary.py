"""dictionary module for Hyperactive optimization."""

# Email: simon.blanke@yahoo.com
# License: MIT License


class DictClass:
    """DictClass class."""

    def __init__(self):
        self.para_dict = {}

    def __getitem__(self, key):
        """Get item from parameter dictionary."""
        return self.para_dict[key]

    def keys(self):
        """Keys function."""
        return self.para_dict.keys()

    def values(self):
        """Values function."""
        return self.para_dict.values()
