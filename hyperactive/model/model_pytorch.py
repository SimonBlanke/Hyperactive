# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from sklearn.model_selection import train_test_split
from importlib import import_module

from .model_keras import KerasModel


class PytorchModel(KerasModel):
    def __init__(self, _config_):
        super().__init__(_config_)
        self.search_config = _config_.search_config

    def _create_model(self, layers_para_dict):

        from torch.nn import Sequential

        model = Sequential()

        for i, layer_key in enumerate(self.layerStr_2_kerasLayer_dict.keys()):
            layer = self.layerStr_2_kerasLayer_dict[layer_key]

            print(
                "layer(**layers_para_dict[layer_key])",
                layer(**layers_para_dict[layer_key]),
            )

            model.add_module(str(i), layer(**layers_para_dict[layer_key]))

        return model

    def set_optimizer(self, optim_str):
        module = import_module("torch.optim")
        return getattr(module, optim_str)

    def set_loss(self, loss_str):
        module = import_module("torch.nn")
        return getattr(module, loss_str)

    def _compile(self, compile_para_dict):
        self.optim = self.set_optimizer(compile_para_dict["optimizer"])
        self.loss = self.set_loss(compile_para_dict["loss"])

    def _fit(self, model, fit_para_dict):
        model.train(True)

        epochs = fit_para_dict["epochs"]
        batch_size = fit_para_dict["batch_size"]

        from torch.utils import data

        training_set = Dataset(partition["train"], labels)
        training_generator = data.DataLoader(training_set, **params)

        for epoch in epochs:
            for i, data in enumerate(trainloader, 0):
                X_train, y_true = data

                self.optim.zero_grad()
                y_pred = model(X_train)
                loss = self.loss(y_pred, y_true)
                loss.backward()
                self.optim.step()

    def train_model(self, keras_para_dict, X, y):
        layers_para_dict = self._trafo_hyperpara_dict(keras_para_dict)
        model = self._create_model(layers_para_dict)

        if list(layers_para_dict.keys())[0] == "torch.compile.0":
            compile_para_dict = self._get_compile_parameter(layers_para_dict)

        if list(layers_para_dict.keys())[1] == "torch.fit.0":
            fit_para_dict = self._get_fit_parameter(layers_para_dict)

        del layers_para_dict["torch.compile.0"]
        del layers_para_dict["torch.fit.0"]

        self._compile(compile_para_dict)
        self._fit(model, fit_para_dict)

        # model.compile(**compile_para_dict)

        if self.cv > 1:
            score = self._cross_val_keras(model, X, y, fit_para_dict)
        elif self.cv < 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=self.cv
            )

            fit_para_dict["x"] = X_train
            fit_para_dict["y"] = y_train
            model.fit(**fit_para_dict)
            y_pred = model.predict(X_test)
            score = self.metric_(y_test, y_pred)
        else:
            score = 0
            model.fit(X, y)

        if self.metric_type == "score":
            return score, model
        elif self.metric_type == "loss":
            return -score, model
