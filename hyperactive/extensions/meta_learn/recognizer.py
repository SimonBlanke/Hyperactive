# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import itertools
import pandas as pd

from .collector._dataset_features import DatasetFeatures


def expand_dataframe(data_pd, length):
    data_pd = pd.DataFrame(data_pd, index=range(length))
    columns = data_pd.columns
    for column in columns:
        data_pd[column] = data_pd[column][0]

    return data_pd


def merge_meta_data(features_from_dataset, features_from_model):
    length = len(features_from_model)
    features_from_dataset = expand_dataframe(features_from_dataset, length)

    features_from_dataset = features_from_dataset.reset_index()
    features_from_model = features_from_model.reset_index()

    if "index" in features_from_dataset.columns:
        features_from_dataset = features_from_dataset.drop("index", axis=1)
    if "index" in features_from_model.columns:
        features_from_model = features_from_model.drop("index", axis=1)

    all_features = pd.concat(
        [features_from_dataset, features_from_model], axis=1, ignore_index=False
    )

    return all_features


class Recognizer:
    def __init__(self, _cand_):
        self.search_config = _cand_.search_config

        self.model_list = list(self.search_config.keys())
        self.model_name = self.model_list[0]
        self.search_space = self.search_config[self.model_name]

    def get_test_metadata(self, data_train):
        self.collector_dataset = DatasetFeatures()

        md_dataset = self.collector_dataset.collect(data_train)
        md_model = self._features_from_model()

        X_test = merge_meta_data(md_dataset, md_model)

        return X_test

    def _features_from_model(self):
        keys, values = zip(*self.search_space.items())
        meta_reg_input = [dict(zip(keys, v)) for v in itertools.product(*values)]

        md_model = pd.DataFrame(meta_reg_input)

        return md_model
