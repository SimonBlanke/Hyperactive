# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from pathlib import Path
from sklearn.externals import joblib

from .meta_regressor import MetaRegressor
from .label_encoder import label_encoder_dict


class Predictor:
    def __init__(self, search_config, meta_regressor_path):
        self.search_config = search_config
        self.meta_regressor_path = meta_regressor_path
        # self.meta_regressor = MetaRegressor()

        self.model_list = list(self.search_config.keys())
        self.model_name = self.model_list[0]

        # self._get_hyperpara()

        self.meta_regressor = None

    def search(self, X_test):
        self.meta_regressor = self._load_model()

        best_hyperpara_dict, best_score = self._predict(X_test)
        return best_hyperpara_dict, best_score

    def _load_model(self):
        path = self._get_meta_regressor_path()

        if Path(self.meta_regressor_path).exists():
            reg = joblib.load(path)
            return reg
        else:
            print("No proper meta regressor found\n")

    def _get_meta_regressor_path(self):
        model_key = self.search_config.keys()
        path = self.meta_regressor_path + str(*model_key) + ":meta_regressor.pkl"
        print("Load meta regressor from" + path)

        return path

    def _predict(self, X_test):
        X_test = self._label_enconding(X_test)
        # print(X_test.info())
        score_pred = self.meta_regressor.predict(X_test)

        best_features, best_score = self._find_best_hyperpara(X_test, score_pred)

        list1 = list(self.search_config[str(*self.search_config.keys())].keys())

        keys = list(best_features[list1].columns)
        values = list(*best_features[list1].values)
        best_hyperpara_dict = dict(zip(keys, values))

        best_hyperpara_dict = self._decode_hyperpara_dict(best_hyperpara_dict)

        return best_hyperpara_dict, best_score

    def _decode_hyperpara_dict(self, hyperpara_dict):
        for hyperpara_key in label_encoder_dict[self.model_name]:
            if hyperpara_key in hyperpara_dict:
                inv_label_encoder_dict = {
                    v: k
                    for k, v in label_encoder_dict[self.model_name][
                        hyperpara_key
                    ].items()
                }

                encoded_values = hyperpara_dict[hyperpara_key]

                hyperpara_dict[hyperpara_key] = inv_label_encoder_dict[encoded_values]

        return hyperpara_dict

    def _find_best_hyperpara(self, features, scores):
        N_best_features = 1

        scores = np.array(scores)
        index_best_scores = list(scores.argsort()[-N_best_features:][::-1])

        best_score = scores[index_best_scores][0]
        best_features = features.iloc[index_best_scores]

        return best_features, best_score

    def _get_hyperpara(self):
        self.hyperpara_dict = label_encoder_dict[self.model_name]

    def _label_enconding(self, X_train):
        for hyperpara_key in self.hyperpara_dict:
            to_replace = {hyperpara_key: self.hyperpara_dict[hyperpara_key]}
            X_train = X_train.replace(to_replace)
        X_train = X_train.convert_objects()

        return X_train
