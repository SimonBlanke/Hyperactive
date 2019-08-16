# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


class MetaRegressor(object):
    def __init__(self, meta_learn_path=None):
        self.meta_reg = None
        self.score_col_name = "mean_test_score"

        self.path_name = meta_learn_path
        self.path_meta_data = self.path_name + "/meta_data/"

    def train_meta_regressor(self, model_list):
        for model_str in model_list:
            X_train, y_train = self._read_meta_data(model_str)

            self._train_regressor(X_train, y_train)
            self._store_model(model_str)

    def _read_meta_data(self, model_str):
        data_str = self.path_meta_data + model_str + "*.csv"
        metadata_name_list = glob.glob(data_str)

        meta_data_list = []
        for metadata_name in metadata_name_list:
            meta_data = pd.read_csv(metadata_name)
            meta_data_list.append(meta_data)

        meta_data = pd.concat(meta_data_list, ignore_index=True)

        column_names = meta_data.columns
        score_name = [name for name in column_names if self.score_col_name in name]

        X_train = meta_data.drop(score_name, axis=1)
        y_train = meta_data[score_name]

        y_train = self._scale(y_train)

        return X_train, y_train

    def _scale(self, y_train):
        # scale the score -> important for comparison of meta data from datasets in meta regressor training
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train)
        y_train = pd.DataFrame(y_train, columns=["mean_test_score"])

        return y_train

    def _train_regressor(self, X_train, y_train):
        if self.meta_reg is None:
            n_estimators = int(y_train.shape[0] / 50 + 50)

            self.meta_reg = GradientBoostingRegressor(n_estimators=n_estimators)
            self.meta_reg.fit(X_train, y_train)

    def _store_model(self, model_str):
        meta_reg_path = self.path_name + "/meta_regressor/"
        if not os.path.exists(meta_reg_path):
            os.makedirs(meta_reg_path)

        path = meta_reg_path + model_str + "_metaregressor.pkl"
        joblib.dump(self.meta_reg, path)
