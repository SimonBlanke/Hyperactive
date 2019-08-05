# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .candidate_lightgbm import LightGbmCandidate
from ..model import CatBoostModel


class CatBoostCandidate(LightGbmCandidate):
    def __init__(self, nth_process, _config_):
        super().__init__(nth_process, _config_)
        self._model_ = CatBoostModel(_config_, self.search_config_key)
