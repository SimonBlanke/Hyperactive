# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .model_sklearn import ScikitLearnModel


class XGBoostModel(ScikitLearnModel):
    def __init__(self, _config_, search_config_key):
        super().__init__(_config_, search_config_key)

        self.search_config_key = search_config_key
        self.model = self._get_model(search_config_key)
