# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
from sklearn.model_selection import KFold

from .metrics import dl_scores, dl_losses
from .model import Model


class DeepLearner(Model):
    def __init__(self, _config_):
        super().__init__(_config_)
        self.search_config = _config_.search_config

        # if no metric was passed
        if isinstance(self.metric, str):
            self.metric = [self.metric]

        self.layerStr_2_kerasLayer_dict = self._layer_dict(_config_.search_config)
        # self.n_layer = len(self.layerStr_2_kerasLayer_dict.keys())

        self._get_search_config_onlyLayers()

        self.scores = dl_scores
        self.losses = dl_losses

        self._get_metric_type_keras()

    def _get_metric_type_keras(self):
        if self.metric[0] in self.scores:
            self.metric_type = "score"
        elif self.metric[0] in self.losses:
            self.metric_type = "loss"

    def _get_search_config_onlyLayers(self):
        self.search_config_onlyLayers = dict(self.search_config)
        if list(self.search_config.keys())[0] == "keras.compile.0":
            del self.search_config_onlyLayers["keras.compile.0"]

            if list(self.search_config.keys())[1] == "keras.fit.0":
                del self.search_config_onlyLayers["keras.fit.0"]

        # elif list(self.search_config.keys())[0] == "keras.fit.0":
        #     del self.search_config_onlyLayers["keras.fit.0"]

    def _layer_dict(self, search_config):
        layerStr_2_kerasLayer_dict = {}

        for layer_key in search_config.keys():
            layer_str, nr = layer_key.rsplit(".", 1)

            # nr=0 are compile and fit _get_fit_parameter
            if nr != "0":
                layer = self._get_model(layer_str)
                layerStr_2_kerasLayer_dict[layer_key] = layer

        return layerStr_2_kerasLayer_dict

    def trafo_hyperpara_dict_lists(self, keras_para_dict):
        layers_para_dict = {}

        for layer_str_1 in list(self.search_config.keys()):

            layer_para_dict = {}
            for layer_key in keras_para_dict.keys():
                layer_str_2, para = layer_key.rsplit(".", 1)

                if layer_str_1 == layer_str_2:
                    layer_para_dict[para] = [keras_para_dict[layer_key]]

            layers_para_dict[layer_str_1] = layer_para_dict

        return layers_para_dict

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

    def _create_model(self, layers_para_dict):

        from keras.models import Sequential

        model = Sequential()

        for layer_key in self.layerStr_2_kerasLayer_dict.keys():
            layer = self.layerStr_2_kerasLayer_dict[layer_key]

            model.add(layer(**layers_para_dict[layer_key]))

        return model

    def _get_compile_parameter(self, layers_para_dict):
        compile_para_dict = layers_para_dict[list(layers_para_dict.keys())[0]]

        return compile_para_dict

    def _get_fit_parameter(self, layers_para_dict):
        fit_para_dict = layers_para_dict[list(layers_para_dict.keys())[1]]

        return fit_para_dict

    def _cross_val_keras(self, model, X, y):
        scores = []

        kf = KFold(n_splits=self.cv, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            score = model.evaluate(X_test, y_test)[1]
            scores.append(score)

        return np.array(scores).mean()

    def _train_split(self, model, fit_para_dict, X, y):
        model.fit(**fit_para_dict)
        return model.evaluate(X, y)[1], model

    def train_model(self, keras_para_dict, X, y):
        layers_para_dict = self._trafo_hyperpara_dict(keras_para_dict)
        model = self._create_model(layers_para_dict)

        if list(layers_para_dict.keys())[0] == "keras.compile.0":
            compile_para_dict = self._get_compile_parameter(layers_para_dict)
        # else:
        #     compile_para_dict = {"loss": "binary_crossentropy", "optimizer": "SGD"}
        if list(layers_para_dict.keys())[1] == "keras.fit.0":
            fit_para_dict = self._get_fit_parameter(layers_para_dict)
        # else:
        #     fit_para_dict = {"epochs": 10, "batch_size": 100}

        del layers_para_dict["keras.compile.0"]
        del layers_para_dict["keras.fit.0"]

        compile_para_dict["metrics"] = self.metric
        fit_para_dict["x"] = X
        fit_para_dict["y"] = y

        model.compile(**compile_para_dict)

        if self.cv > 1:
            score = self._cross_val_keras(model, X, y)
        elif self.cv < 1:
            fit_para_dict["validation_split"] = self.cv
            score, model = self._train_split(model, fit_para_dict, X, y)
        else:
            score, model = self._train_split(model, fit_para_dict, X, y)

        if self.metric_type == "score":
            return score, model
        elif self.metric_type == "loss":
            return -score, model
