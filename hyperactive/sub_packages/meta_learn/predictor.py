# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from pathlib import Path
from sklearn.externals import joblib

from .label_encoder import label_encoder_dict
from .data_wrangler import find_best_hyperpara


class Predictor:
    def __init__(self, search_config, meta_reg_path):
        self.search_config = search_config
        self.meta_reg_path = meta_reg_path

        self.model_list = list(self.search_config.keys())
        self.model_str = self.model_list[0]

        self.para = label_encoder_dict[self.model_str]

        self.meta_reg = None

    def search(self, X_test):
        self.meta_reg = self._load_model()

        best_para, best_score = self._predict(X_test)
        return best_para, best_score

    def _load_model(self):
        path = self._get_meta_regressor_path()

        if Path(self.meta_reg_path).exists():
            reg = joblib.load(path)
            return reg
        else:
            print("No proper meta regressor found\n")

    def _get_meta_regressor_path(self):
        model_key = self.search_config.keys()
        path = self.meta_reg_path + str(*model_key) + "_metaregressor.pkl"
        print("Load meta regressor from" + path)

        return path

    def _predict(self, X_test):
        X_test = self._label_enconding(X_test)
        # print(X_test.info())
        score_pred = self.meta_reg.predict(X_test)

        best_features, best_score = find_best_hyperpara(X_test, score_pred)

        list1 = list(self.search_config[str(*self.search_config.keys())].keys())

        keys = list(best_features[list1].columns)
        values = list(*best_features[list1].values)
        best_para = dict(zip(keys, values))

        best_para = self._decode_hyperpara_dict(best_para)

        return best_para, best_score

    def _decode_hyperpara_dict(self, para):
        for hyperpara_key in label_encoder_dict[self.model_str]:
            if hyperpara_key in para:
                inv_label_encoder_dict = {
                    v: k
                    for k, v in label_encoder_dict[self.model_str][
                        hyperpara_key
                    ].items()
                }

                encoded_values = para[hyperpara_key]
                para[hyperpara_key] = inv_label_encoder_dict[encoded_values]

        return para

    def _label_enconding(self, X_train):
        for hyperpara_key in self.para:
            to_replace = {hyperpara_key: self.para[hyperpara_key]}
            X_train = X_train.replace(to_replace)
        X_train = X_train.infer_objects()

        return X_train
