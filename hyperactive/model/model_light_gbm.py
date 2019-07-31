# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .metrics import ml_scores, ml_losses
from .model import Model


class LightGbmModel(Model):
    def __init__(self, _config_, search_config_key):
        super().__init__(_config_)

        self.scores = ml_scores
        self.losses = ml_losses
