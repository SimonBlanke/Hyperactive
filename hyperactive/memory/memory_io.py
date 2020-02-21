# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

from .util import get_model_id, _hash2obj
from .paths import (
    get_meta_path,
    get_model_path,
    get_date_path,
    get_datetime,
    get_meta_data_name,
)


class MemoryIO:
    def __init__(self, _space_, _main_args_, _cand_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.meta_data_name = get_meta_data_name(_main_args_.X, _main_args_.y)
        self.score_col_name = "_score_"

        model_id = get_model_id(_cand_.func_)
        self.datetime = get_datetime()

        self.meta_path = get_meta_path()
        self.model_path = self.meta_path + get_model_path(model_id)
        self.date_path = self.model_path + get_date_path(self.datetime)

        self.dataset_info_path = self.model_path + "dataset_info/"

        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path, exist_ok=True)

        self.hash2obj = _hash2obj(_space_.search_space, self.model_path)
