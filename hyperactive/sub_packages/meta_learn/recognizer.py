# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import itertools
import pandas as pd


from importlib import import_module

from .collector import Collector
from .label_encoder import label_encoder_dict
from ..insight import Insight


class Recognizer:
    def __init__(self, search_config):
        self.path = None
        self.search_config = search_config
        self.meta_regressor = None
        self.meta_knowledge = None

        self.model_name = None
        self.search_space = None

        self.model = None

        self.data_name = None
        self.data_test = None

        self.hyperpara_dict = None

        self.collector = Collector(search_config)

        self.model_list = list(self.search_config.keys())

        self.model_name = self.model_list[0]
        self.search_space = self.search_config[self.model_name]

    def get_test_metadata(self, data_train):
        self.insight = Insight(data_train[0], data_train[1])

        features_from_dataset = self.insight.collect(self.model_name, data_train)

        self.hyperpara_dict = self._get_hyperpara(self.model_name)

        model = self._import_model(self.model_name)
        self.model = model()

        features_from_model = self._features_from_model()

        X_test = self.collector._merge_data(features_from_dataset, features_from_model)

        return X_test

    def _import_model(self, model):
        sklearn, submod_func = model.rsplit(".", 1)
        module = import_module(sklearn)
        model = getattr(module, submod_func)

        return model

    def _get_hyperpara(self, model_name):
        return label_encoder_dict[model_name]

    def _features_from_model(self):
        keys, values = zip(*self.search_space.items())
        meta_reg_input = [dict(zip(keys, v)) for v in itertools.product(*values)]

        features_from_model = pd.DataFrame(meta_reg_input)

        default_hyperpara_df = self.collector.dataCollector_model._get_default_hyperpara(
            self.model, len(features_from_model)
        )
        features_from_model = self.collector.dataCollector_model._merge_dict(
            features_from_model, default_hyperpara_df
        )
        features_from_model = features_from_model.reindex(
            sorted(features_from_model.columns), axis=1
        )

        return features_from_model
