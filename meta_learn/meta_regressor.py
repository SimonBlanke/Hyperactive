# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import pandas as pd

# import xgboost as xgb


from sklearn.ensemble import GradientBoostingRegressor

from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler


class MetaRegressor(object):
    def __init__(self, model_name):
        self.path = "./data/meta_knowledge"
        self.meta_regressor = None

        self.model_name = model_name
        # self._get_model_name()

    def train_meta_regressor(self):
        X_train, y_train = self._get_meta_knowledge()

        # X_train = self._label_enconding(X_train)
        # print(X_train)
        self._train_regressor(X_train, y_train)
        self._store_model()

    def _get_model_name(self):
        model_name = self.path.split("/")[-1]
        return model_name

    def _get_meta_knowledge(self):
        # data = pd.read_csv(self.path)
        data = pd.read_csv(self.path)

        column_names = data.columns
        score_name = [name for name in column_names if "mean_test_score" in name]

        X_train = data.drop(score_name, axis=1)
        y_train = data[score_name]

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
            self.meta_regressor.fit(X_train, y_train)
            print("time: ", round((time.time() - time1), 4))

    def _store_model(self):
        filename = "./data/" + str(self.model_name) + "_meta_regressor.pkl"
        # print(filename)
        joblib.dump(self.meta_regressor, filename)
