# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time

from importlib import import_module
from sklearn.model_selection import cross_val_score


class Model:
    def __init__(self, search_config, metric, cv):
        self.search_config = search_config
        self.metric = metric
        self.cv = cv

    def _get_model(self, model):
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model


class DeepLearner(Model):
    def __init__(self, search_config, metric, cv):
        super().__init__(search_config, metric, cv)

        self.search_config = search_config
        self.metric = metric
        self.cv = cv

        # if no metric was passed
        if isinstance(self.metric, str):
            self.metric = [self.metric]

        self.layerStr_2_kerasLayer_dict = self._layer_dict(search_config)
        # self.n_layer = len(self.layerStr_2_kerasLayer_dict.keys())

        self._get_search_config_onlyLayers()

        self.scores = [
            "accuracy",
            "binary_accuracy",
            "categorical_accuracy",
            "sparse_categorical_accuracy",
            "top_k_categorical_accuracy",
            "sparse_top_k_categorical_accuracy",
        ]

        self.losses = [
            "mean_squared_error",
            "mean_absolute_error",
            "mean_absolute_percentage_error",
            "mean_squared_logarithmic_error",
            "squared_hinge",
            "hinge",
            "categorical_hinge",
            "logcosh",
            "categorical_crossentropy",
            "sparse_categorical_crossentropy",
            "binary_crossentropy",
            "kullback_leibler_divergence",
            "poisson",
            "cosine_proximity",
        ]

        self._get_metric_type_keras()

    def _get_metric_type_keras(self):
        if self.metric[0] in self.scores:
            self.metric_type = "score"
        elif self.metric[0] in self.losses:
            self.metric_type = "loss"
        else:
            raise ValueError(
                "\n", self.metric, "not in list of compatible scoring functions"
            )

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

    def _create_keras_model(self, keras_para_dict):
        from keras.models import Sequential

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

        compile_para_dict["metrics"] = self.metric
        fit_para_dict["x"] = X_train
        fit_para_dict["y"] = y_train

        model.compile(**compile_para_dict)
        model.fit(**fit_para_dict)

        score = model.evaluate(X_train, y_train)[1]

        if self.metric_type == "score":
            return score, 0, model
        elif self.metric_type == "loss":
            return -score, 0, model


class MachineLearner(Model):
    def __init__(self, search_config, metric, cv, model_str):
        super().__init__(search_config, metric, cv)

        self.search_config = search_config
        self.metric = metric
        self.cv = cv
        self.model_str = model_str

        self.model = self._get_model(model_str)

        self.scores = [
            "accuracy",
            "balanced_accuracy",
            "average_precision",
            "f1",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "precision",
            "recall",
            "jaccard",
            "roc_auc",
            "explained_variance",
            "r2",
        ]

        self.losses = [
            "brier_score_loss",
            "neg_log_loss",
            "max_error",
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
        ]

        self._get_metric_type_sklearn()

    def _get_metric_type_sklearn(self):
        if self.metric in self.scores:
            self.metric_type = "score"
        elif self.metric in self.losses:
            self.metric_type = "loss"
        else:
            raise ValueError("\n Metric not compatible with sklearn scoring functions")

    def create_start_point(self, sklearn_para_dict, nth_process):
        start_point = {}
        model_str = self.model_str + "." + str(nth_process)

        temp_dict = {}
        for para_key in sklearn_para_dict:
            temp_dict[para_key] = [sklearn_para_dict[para_key]]

        start_point[model_str] = temp_dict

        return start_point

    def train_model(self, sklearn_para_dict, X_train, y_train):
        sklearn_model = self.model(**sklearn_para_dict)

        time_temp = time.perf_counter()
        scores = cross_val_score(
            sklearn_model, X_train, y_train, scoring=self.metric, cv=self.cv
        )
        train_time = (time.perf_counter() - time_temp) / self.cv

        if self.metric_type == "score":
            return scores.mean(), train_time, sklearn_model
        elif self.metric_type == "loss":
            return -scores.mean(), train_time, sklearn_model
