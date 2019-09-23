# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class Model:
    def __init__(self, _config_):
        self.func_ = list(_config_.search_config.keys())[0]

    def train_model(self, keras_para_dict, X, y):
        score = self.func_(keras_para_dict, X, y)

        return score
