# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def get_dataset_features(X, y):
    data = {
        "X_size": X.size,
        "X_byte_size": X.itemsize,
        "X_ndim": X.ndim,
        "y_size": y.size,
        "y_byte_size": y.itemsize,
        "y_ndim": y.ndim,
    }

    return data
