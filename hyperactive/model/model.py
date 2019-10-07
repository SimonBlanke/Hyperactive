# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


class Model:
    def __init__(self, _config_):
        self.func_ = list(_config_.search_config.keys())[0]

    def train_model(self, keras_para_dict, X, y):
        result = self.func_(keras_para_dict, X, y)
        
        if isinstance(result, tuple):
            score = result[0]
            model = result[1]
        elif isinstance(result, float):
            score = result
            model = None
        else:
            print("Error: model function must return float or tuple")

        return score, model
