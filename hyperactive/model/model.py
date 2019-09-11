# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from importlib import import_module


class Model:
    def __init__(self, _config_):
        self.metric = _config_.metric
        self.cv = _config_.cv

        self.func_ = list(_config_.search_config.keys())[0]

    def train_model(self, keras_para_dict, X, y):
        score, model = self.func_(keras_para_dict, X, y)

        return score, model

    def _get_model(self, model):
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model
