# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import dill


from .util import get_hash, get_model_id
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

        self.hash2obj = self._hash2obj()

    def _read_dill(self, value):
        paths = self._get_pkl_hash(value)
        for path in paths:
            with open(path, "rb") as fp:
                value = dill.load(fp)
                value = dill.loads(value)
                break

        return value

    def _get_pkl_hash(self, hash):
        paths = glob.glob(self.model_path + hash + "*.pkl")

        return paths

    def _hash2obj(self):
        hash2obj_dict = {}
        para_hash_list = self._get_para_hash_list()

        for para_hash in para_hash_list:
            obj = self._read_dill(para_hash)
            hash2obj_dict[para_hash] = obj

        return hash2obj_dict

    def _get_para_hash_list(self):
        para_hash_list = []
        for key in self._space_.search_space.keys():
            values = self._space_.search_space[key]

            for value in values:
                if (
                    not isinstance(value, int)
                    and not isinstance(value, float)
                    and not isinstance(value, str)
                ):

                    para_dill = dill.dumps(value)
                    para_hash = get_hash(para_dill)
                    para_hash_list.append(para_hash)

        return para_hash_list
