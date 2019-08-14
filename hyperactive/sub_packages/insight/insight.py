# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd
import hashlib
from importlib import import_module

from .dataset_features import get_number_of_instances
from .dataset_features import get_number_of_features
from .dataset_features import get_default_score


class Insight:
    get_number_of_instances = get_number_of_instances
    get_number_of_features = get_number_of_features
    get_default_score = get_default_score

    def __init__(self, X, y):

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(y, pd.core.frame.DataFrame):
            y = y.values

        self.X = X
        self.y = y
        self.data_type = None

    def recognize_data(self):
        x_hash = self._get_hash(self.X)
        # if hash recog fails: try to recog data by other properties

        return x_hash

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_data_type(self):
        self.n_samples = self.X.shape[0]

        if len(self.X.shape) == 2:
            self.data_type = "tabular"
            self.n_features = self.X.shape[1]
        elif len(self.X.shape) == 3:
            self.data_type = "image2D"
            self.n_pixels_x = self.X.shape[1]
            self.n_pixels_y = self.X.shape[2]
        elif len(self.X.shape) == 4:
            self.data_type = "image3D"
            self.n_pixels_x = self.X.shape[1]
            self.n_pixels_y = self.X.shape[2]
            self.n_colors = self.X.shape[3]
        else:
            self.data_type = "sequential"

    def _get_data_complexity(self):
        pass

    def collect(self, model_name, data_train):
        self.data_name = self.recognize_data()
        self.X_train = data_train[0]
        self.y_train = data_train[1]

        # for later use in functions in func_list
        model = self._import_model(model_name)
        self.model = model()

        # List of functions to get the different features of the dataset
        func_list = [
            self.get_number_of_instances,
            self.get_number_of_features,
            self.get_default_score,
        ]

        features_from_dataset = {}
        for func in func_list:
            name, value = func()
            features_from_dataset[name] = value

        features_from_dataset = pd.DataFrame(features_from_dataset, index=[0])
        features_from_dataset = features_from_dataset.reindex(
            sorted(features_from_dataset.columns), axis=1
        )

        return features_from_dataset

    def _import_model(self, model):
        sklearn, submod_func = model.rsplit(".", 1)
        module = import_module(sklearn)
        model = getattr(module, submod_func)

        return model
