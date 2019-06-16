# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from pathlib import Path
from sklearn.externals import joblib

from .meta_regressor import MetaRegressor
from .label_encoder_dict import label_encoder_dict


class Predictor(MetaRegressor):
    def __init__(self, search_config):
        pass

    def _load_model(self):
        path = self._get_meta_regressor_path()

        if Path(self.path).exists():
            reg = joblib.load(path)
            self.meta_regressor = reg
        else:
            print("No proper meta regressor found\n")

    def _get_meta_regressor_path(self):
        model_key = self.search_config.keys()
        path = self.path + str(*model_key) + "_meta_regressor.pkl"
        print("Load meta regressor from" + path)

        return path

    def _predict(self):
        self.meta_knowledge = self._label_enconding(self.meta_knowledge)
        print(self.meta_knowledge)
        print(self.meta_knowledge.info())
        score_pred = self.meta_regressor.predict(self.meta_knowledge)

        best_features, best_score = self._find_best_hyperpara(
            self.meta_knowledge, score_pred
        )

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

    def _label_enconding(self, X_train):
        for hyperpara_key in self.hyperpara_dict:
            to_replace = {hyperpara_key: self.hyperpara_dict[hyperpara_key]}
            X_train = X_train.replace(to_replace)
        X_train = X_train.convert_objects()

        return X_train
