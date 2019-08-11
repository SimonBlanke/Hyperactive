# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .candidate_keras import KerasCandidate
from ..model import PytorchModel


class PytorchCandidate(KerasCandidate):
    def __init__(self, nth_process, _config_):
        super().__init__(nth_process, _config_)
        self._model_ = PytorchModel(_config_)
