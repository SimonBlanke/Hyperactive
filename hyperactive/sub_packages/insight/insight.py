# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd
import hashlib


class Insight:
    def __init__(self):
        pass

    def recognize_data(self, X, y):
        x_hash = self._get_hash(X)
        # if hash recog fails: try to recog data by other properties

        return x_hash

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()
