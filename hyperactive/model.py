# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from importlib import import_module
from sklearn.model_selection import cross_val_score
from keras.models import Sequential


class Model:
    def __init__(self, search_config, scoring, cv):
        self.search_config = search_config
        self.scoring = scoring
        self.cv = cv

    def _get_model(self, model):
        sklearn_str, model_str = model.rsplit(".", 1)
        module = import_module(sklearn_str)
        model = getattr(module, model_str)

        return model


class DeepLearner(Model):
    def __init__(self, search_config, scoring, cv):
        super().__init__(search_config, scoring, cv)

        self.search_config = search_config
        self.scoring = scoring
        self.cv = cv

        self.layer_dict = self._layer_dict(search_config)
        self.n_layer = len(self.layer_dict.keys())

    def _layer_dict(self, search_config):
        layer_dict = {}
        for layer_key in search_config.keys():
            layer_str, nr = layer_key.rsplit(".", 1)

            layer = self._get_model(layer_str)
            layer_dict[layer_key] = layer

        return layer_dict

    def _trafo_hyperpara_dict(self, keras_para_dict):
        self.layer_para_dict = {}

        nr_str_temp = "1"
        self.layer_para_dict_temp = {}
        for layer_key in keras_para_dict.keys():
            layer_nr_str, para_str = layer_key.rsplit(".", 1)
            layer_str, nr_str = layer_nr_str.rsplit(".", 1)

            if nr_str_temp != nr_str:
                self.layer_para_dict[self.layer_nr_str_temp] = self.layer_para_dict_temp
                self.layer_para_dict_temp = {}

            self.layer_para_dict_temp[para_str] = keras_para_dict[layer_key]

            self.layer_nr_str_temp = layer_nr_str
            nr_str_temp = nr_str

        self.layer_para_dict[self.layer_nr_str_temp] = self.layer_para_dict_temp

    def _create_keras_model(self, keras_para_dict):
        self._trafo_hyperpara_dict(keras_para_dict)

        model = Sequential()
        for layer_key in self.layer_dict.keys():
            layer = self.layer_dict[layer_key]

            model.add(layer(**self.layer_para_dict[layer_key]))

        return model

    def train_model(self, keras_para_dict, X_train, y_train):
        model = self._create_keras_model(keras_para_dict)

        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        model.fit(X_train, y_train, epochs=1, batch_size=100, verbose=1)

        if self.cv == 1:
            score = model.evaluate(X_train, y_train)[1]

        return score, time, model


class MachineLearner(Model):
    def __init__(self, search_config, scoring, cv):
        super().__init__(search_config, scoring, cv)

        self.search_config = search_config
        self.scoring = scoring
        self.cv = cv

        self.model_key = list(self.search_config.keys())[0]
        self.model = self._get_model(self.model_key)

    def _create_sklearn_model(self, sklearn_para_dict):
        return self.model(**sklearn_para_dict)

    def train_model(self, sklearn_para_dict, X_train, y_train):
        sklearn_model = self._create_sklearn_model(sklearn_para_dict)

        time_temp = time.perf_counter()
        scores = cross_val_score(
            sklearn_model, X_train, y_train, scoring=self.scoring, cv=self.cv
        )
        train_time = (time.perf_counter() - time_temp) / self.cv

        return scores.mean(), train_time, sklearn_model
