# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import pandas as pd
import hashlib


class Insight:
    def __init__(self):
        pass

    def recognize_data(self, X, y):
        x_hash = self._get_hash(X)
        # if hash recog fails: try to recog data by other properties

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()


data_hash_dict = {
    "1c7f717bc3f902d5b72ba6735900b3454a0b96fa": "dict_path",
    "2c7f717bc3f902d5b72ba6735900b3454a0b96fa": "dict_path",
    "3c7f717bc3f902d5b72ba6735900b3454a0b96fa": "dict_path",
}

model_dict = {
    "sklearn.ensemble.GradientBoostingClassifier": "dict_path",
    "xgboost.XGBClassifier": "dict_path",
    "torch": "dict_path",
}

    def _bla(self, data_hash, model_str):
        meta_data_path = './meta_data/'+data_hash+'/'+model_str+'.csv'

        meta_data = pd.to_csv(meta_data_path)


data = pd.DataFrame(data, index="para_pos_string1", columns=[0.1, 0.2, ..., 10])
