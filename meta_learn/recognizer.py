# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import itertools
import pandas as pd

from tqdm import tqdm


from importlib import import_module

from .memory import Memory
from .label_encoder_dict import label_encoder_dict


class Recognizer(Memory):
    def __init__(self, search_config):
        self.path = None
        self.search_config = search_config
        self.meta_regressor = None
        self.meta_knowledge = None

        self.model_name = None
        self.search_space = None

        self.model = None
        self.X_train = None
        self.y_train = None

        self.data_name = None
        self.data_test = None

        self.hyperpara_dict = None

        self.dataCollector_model = DataCollector_model(search_config)

        current_path = os.path.realpath(__file__)
        self.path_name, file_name = current_path.rsplit("/", 1)

        self.path = self.path_name + "/meta_regressor/"

    def search_optimum(self, data_dict):
        for data_name, data_test in tqdm(data_dict.items()):
            self.data_name = data_name
            self.data_test = data_test

            self.X_train = data_test[0]
            self.y_train = data_test[1]

            self._load_model()

            if not self.meta_regressor:
                print("Error: No meta regressor loaded\n")
                return 0

            self._search(self.X_train)
            hyperpara_dict, best_score = self._predict()

            return hyperpara_dict, best_score

    def _search(self, X_train):
        for model_key in self.search_config.keys():
            self.model_name = model_key

            self.hyperpara_dict = self._get_hyperpara(model_key)

            model = self._import_model(model_key)
            self.model = model()

            self.search_space = self.search_config[model_key]

            if len(self.search_space.items()) == 0:
                print("Error: hyperparameter search dict is empty\n")
                break

            features_from_model = self._features_from_model()

            dataCollector_dataset = DataCollector_dataset(self.search_config)
            features_from_dataset = dataCollector_dataset.collect(
                self.data_name, self.data_test
            )

            features_from_dataset = pd.DataFrame(
                features_from_dataset, index=range(len(features_from_model))
            )

            columns = features_from_dataset.columns
            for column in columns:
                features_from_dataset[column] = features_from_dataset[column][0]

            self.meta_knowledge = self._concat_dataframes(
                features_from_dataset, features_from_model
            )

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

        default_hyperpara_df = self.dataCollector_model._get_default_hyperpara(
            self.model, len(features_from_model)
        )
        features_from_model = self.dataCollector_model._merge_dict(
            features_from_model, default_hyperpara_df
        )
        features_from_model = features_from_model.reindex_axis(
            sorted(features_from_model.columns), axis=1
        )

        return features_from_model
