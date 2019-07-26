# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from importlib import import_module


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
