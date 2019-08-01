# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .candidate import Candidate
from ..model import ScikitLearnModel
from ..init_position import InitMLSearchPosition


class ScikitLearnCandidate(Candidate):
    def __init__(self, nth_process, _config_):
        super().__init__(nth_process, _config_)

        self.nth_process = nth_process
        self.search_config_key = _config_._get_sklearn_model(nth_process)

        self._space_.create_mlSearchSpace(self.search_config_key)
        self._model_ = ScikitLearnModel(_config_, self.search_config_key)

        self._init_ = InitMLSearchPosition(
            self._space_, self._model_, _config_.warm_start, _config_.scatter_init
        )

    def create_start_point(self, para):
        start_point = {}
        model_str = self.search_config_key + "." + str(self.nth_process)

        temp_dict = {}
        for para_key in para:
            temp_dict[para_key] = [para[para_key]]

        start_point[model_str] = temp_dict

        return start_point

    def _get_warm_start(self):
        para_best = self._space_.pos2para(self.pos_best)
        warm_start = self.create_start_point(para_best)

        return warm_start
