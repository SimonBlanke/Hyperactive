# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd


class DatasetFeatures:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def get_features(self):
        features_X_list = [self.X.size, self.X.itemsize, self.X.ndim]
        features_y_list = [self.y.size, self.y.itemsize, self.y.ndim]

        features_np = np.array(features_X_list + features_y_list)

        col_names = [
            "X_size",
            "X_byte_size",
            "X_ndim",
            "y_size",
            "y_byte_size",
            "y_ndim",
        ]

        return pd.DataFrame(features_np, columns=col_names)
