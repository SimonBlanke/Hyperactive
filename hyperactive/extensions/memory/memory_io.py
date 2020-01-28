# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import dill
import datetime
import inspect
import hashlib


class MemoryIO:
    def __init__(self, _space_, _main_args_, _cand_, memory_dict):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.feature_hash = self._get_hash(_main_args_.X)
        self.label_hash = self._get_hash(_main_args_.y)

        self.score_col_name = "score"

        current_path = os.path.realpath(__file__)
        self.meta_learn_path, _ = current_path.rsplit("/", 1)

        func_str = self._get_func_str(_cand_.func_)
        self.func_path_ = self._get_hash(func_str.encode("utf-8")) + "/"

        self.datetime = "run_data/" + datetime.datetime.now().strftime(
            "%d.%m.%Y - %H:%M:%S"
        )

        self.meta_path = self.meta_learn_path + "/meta_data/"
        self.func_path = self.meta_path + self.func_path_
        self.date_path = self.meta_path + self.func_path_ + self.datetime + "/"

        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path, exist_ok=True)

        self.hash2obj = self._hash2obj()

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    """
    def is_sha1(maybe_sha):
        if len(maybe_sha) != 40:
            return False
        try:
            sha_int = int(maybe_sha, 16)
        except ValueError:
            return False
        return True
    """

    def _read_dill(self, value):
        paths = self._get_pkl_hash(value)
        for path in paths:
            with open(path, "rb") as fp:
                value = dill.load(fp)
                value = dill.loads(value)
                break

        return value

    def _get_pkl_hash(self, hash):
        paths = glob.glob(self.func_path + hash + "*.pkl")

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
                    para_hash = self._get_hash(para_dill)
                    para_hash_list.append(para_hash)

        return para_hash_list
