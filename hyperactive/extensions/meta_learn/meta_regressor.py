# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from ._recognizer import Recognizer
from ._predictor import Predictor


class MetaRegressor:
    def __init__(self):
        self.meta_reg = None
        self.score_col_name = "mean_test_score"

    def fit(self, X, y):
        self._train_regressor(X, y)

    def predict(self, X, y):
        self.recognizer = Recognizer(self.search_config)
        self.predictor = Predictor(self.search_config, self.meta_regressor_path)

        X_test = self.recognizer.get_test_metadata([X, y])

        best_hyperpara_dict, best_score = self.predictor.search(X_test)

        return best_hyperpara_dict, best_score

    def _scale(self, y):
        # scale the score -> important for comparison of meta data from datasets in meta regressor training
        scaler = MinMaxScaler()
        y = scaler.fit_transform(y)
        y = pd.DataFrame(y, columns=["mean_test_score"])

        return y

    def _train_regressor(self, X, y):
        if self.meta_reg is None:
            n_estimators = int(y.shape[0] / 50 + 50)

            self.meta_reg = GradientBoostingRegressor(n_estimators=n_estimators)
            self.meta_reg.fit(X, y)

    def store_model(self, path):
        joblib.dump(self.meta_reg, path)

    def load_model(self, path):
        self.meta_reg = joblib.load(path)
