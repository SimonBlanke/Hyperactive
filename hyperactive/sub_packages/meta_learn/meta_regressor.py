# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import time
import pandas as pd

# import xgboost as xgb


from sklearn.ensemble import GradientBoostingRegressor

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


class MetaRegressor(object):
    def __init__(self, meta_learn_path=None):
        self.meta_regressor = None
        self.score_col_name = "mean_test_score"
        # self._get_model_name()

        self.path_name = meta_learn_path
        self.path_meta_data = self.path_name + "/meta_data/"

    def train_meta_regressor(self, model_list, dataset_str):
        for model_str in model_list:
            X_train, y_train = self._read_meta_data(model_str, dataset_str)

            self._train_regressor(X_train, y_train)
            self._store_model(model_str)

    def _get_dataset_name(self, model_str, data_hash):
        file_name = model_str + "___" + data_hash + "___metadata.csv"
        return file_name

    def _get_model_name(self):
        model_name = self.path.split("/")[-1]
        return model_name

    def _read_meta_data(self, model_str, dataset_str):
        # data = pd.read_csv(self.path)
        dataset_name_path = self.path_meta_data + self._get_dataset_name(
            model_str, dataset_str
        )
        meta_data = pd.read_csv(dataset_name_path)

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
        if self.meta_regressor is None:
            n_estimators = int(y_train.shape[0] / 50 + 50)
            print("n_estimators: ", n_estimators)

            time1 = time.time()
            # self.meta_regressor = GradientBoostingRegressor(n_estimators=n_estimators)
            self.meta_regressor = GradientBoostingRegressor(n_estimators=n_estimators)

            print(X_train.columns)
            print("X_train", X_train.head())

            self.meta_regressor.fit(X_train, y_train)
            print("\nmeta_regressor training time: ", round((time.time() - time1), 4))

    def _store_model(self, model_str):
        meta_regressor_path = self.path_name + "/meta_regressor/"
        if not os.path.exists(meta_regressor_path):
            os.makedirs(meta_regressor_path)

        path = meta_regressor_path + model_str + "_metaregressor.pkl"

        joblib.dump(self.meta_regressor, path)
