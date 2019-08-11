# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .model_keras import KerasModel


class PytorchModel(KerasModel):
    def __init__(self, _config_):
        super().__init__(_config_)
