# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class DictClass:
    def __init__(self):
        self.para_dict = {}

    def __getitem__(self, key):
        return self.para_dict[key]

    def keys(self):
        return self.para_dict.keys()

    def values(self):
        return self.para_dict.values()
