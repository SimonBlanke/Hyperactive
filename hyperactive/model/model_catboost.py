# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .model_light_gbm import LightGbmModel


class CatBoostModel(LightGbmModel):
    def __init__(self, _config_, search_config_key):
        super().__init__(_config_, search_config_key)
