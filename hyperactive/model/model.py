# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from importlib import import_module

from .metrics import scores, losses


class Model:
    def __init__(self, _config_):
        self.metric = _config_.metric
        self.cv = _config_.cv

        self.scores = scores
        self.losses = losses

        self._get_metric_type()

    def _get_model(self, model):
        module_str, model_str = model.rsplit(".", 1)
        module = import_module(module_str)
        model = getattr(module, model_str)

        return model

    def _get_metric_type(self):
        if self.metric in self.scores:
            self.metric_type = "score"
        elif self.metric in self.losses:
            self.metric_type = "loss"
