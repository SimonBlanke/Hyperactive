# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .metrics import ml_scores, ml_losses
from .model_sklearn import ScikitLearnModel


class XGBoostModel(ScikitLearnModel):
    def __init__(self, _config_, search_config_key):
        super().__init__(_config_)
        self.search_config_key = search_config_key
        self.model = self._get_model(search_config_key)

        self.scores = ml_scores
        self.losses = ml_losses

        self._get_metric_type_sklearn()
