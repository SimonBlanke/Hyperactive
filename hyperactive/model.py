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

        self.layerStr_2_kerasLayer_dict = self._layer_dict(search_config)
        # self.n_layer = len(self.layerStr_2_kerasLayer_dict.keys())

        self._get_search_config_onlyLayers()

    def _get_search_config_onlyLayers(self):
        self.search_config_onlyLayers = dict(self.search_config)
        if list(self.search_config.keys())[0] == "keras.compile.0":
            del self.search_config_onlyLayers["keras.compile.0"]

            if list(self.search_config.keys())[1] == "keras.fit.0":
                del self.search_config_onlyLayers["keras.fit.0"]

        elif list(self.search_config.keys())[0] == "keras.fit.0":
            del self.search_config_onlyLayers["keras.fit.0"]

    def _get_dict_lengths(self, dict, depth):
        if depth == 0:
            return len(list(dict.keys()))
        elif depth == 1:
            length_list = []

            for key in dict:
                length = len(list(dict[key]))
                length_list.append(length)

            return length_list

    def _layer_dict(self, search_config):
        layerStr_2_kerasLayer_dict = {}

        for layer_key in search_config.keys():
            layer_str, nr = layer_key.rsplit(".", 1)

            # nr=0 are compile and fit _get_fit_parameter
            if nr != "0":
                layer = self._get_model(layer_str)
                layerStr_2_kerasLayer_dict[layer_key] = layer

        return layerStr_2_kerasLayer_dict

    def _trafo_hyperpara_dict(self, keras_para_dict):
        layers_para_dict = {}

        for layer_str_1 in list(self.search_config.keys()):

            layer_para_dict = {}
            for layer_key in keras_para_dict.keys():
                layer_str_2, para = layer_key.rsplit(".", 1)

                if layer_str_1 == layer_str_2:
                    layer_para_dict[para] = keras_para_dict[layer_key]

            layers_para_dict[layer_str_1] = layer_para_dict

        return layers_para_dict

    def _create_keras_model(self, keras_para_dict):
        layers_para_dict = self._trafo_hyperpara_dict(keras_para_dict)

        model = Sequential()
        for layer_key in self.layerStr_2_kerasLayer_dict.keys():
            layer = self.layerStr_2_kerasLayer_dict[layer_key]

            model.add(layer(**layers_para_dict[layer_key]))

        return model, layers_para_dict

    def _get_compile_parameter(self, layers_para_dict):
        compile_para_dict = layers_para_dict[list(layers_para_dict.keys())[0]]

        return compile_para_dict

    def _get_fit_parameter(self, layers_para_dict):
        fit_para_dict = layers_para_dict[list(layers_para_dict.keys())[1]]

        return fit_para_dict

    def train_model(self, keras_para_dict, X_train, y_train):
        model, layers_para_dict = self._create_keras_model(keras_para_dict)

        if list(layers_para_dict.keys())[0] == "keras.compile.0":
            compile_para_dict = self._get_compile_parameter(layers_para_dict)
        else:
            compile_para_dict = {"loss": "binary_crossentropy", "optimizer": "SGD"}
        if list(layers_para_dict.keys())[1] == "keras.fit.0":
            fit_para_dict = self._get_fit_parameter(layers_para_dict)
        else:
            fit_para_dict = {"epochs": 10, "batch_size": 100}

        del layers_para_dict["keras.compile.0"]
        del layers_para_dict["keras.fit.0"]

        compile_para_dict["metrics"] = ["accuracy"]
        fit_para_dict["x"] = X_train
        fit_para_dict["y"] = y_train

        model.compile(**compile_para_dict)
        model.fit(**fit_para_dict)

        # if self.cv == 1:
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
