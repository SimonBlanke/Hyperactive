# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .model import Model


class ChainerModel(Model):
    def __init__(self, _config_):
        super().__init__(_config_)
        self.search_config = _config_.search_config
