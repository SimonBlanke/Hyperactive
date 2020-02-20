# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from .memory_load import MemoryLoad
from .memory_dump import MemoryDump


class BaseMemory:
    def __init__(self, _space_, _main_args_, _cand_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = {}
        self.memory_dict_new = {}

        self.meta_data_found = False

        self.n_dims = None


class ShortTermMemory(BaseMemory):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)


class LongTermMemory(BaseMemory):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)

        self._load_ = MemoryLoad(_space_, _main_args_, _cand_)
        self._dump_ = MemoryDump(_space_, _main_args_, _cand_)

    def load_memory(self, _cand_, _verb_):
        self.memory_dict = self._load_._load_memory(_cand_, _verb_, self.memory_dict)

    def save_memory(self, _main_args_, _opt_args_, _cand_):
        self._dump_._save_memory(_main_args_, _opt_args_, _cand_, self.memory_dict_new)

    def _get_para(self):
        results_dict = self._dump_._get_opt_meta_data(self.memory_dict)

        return pd.DataFrame(results_dict["params"])

    def _get_score(self):
        results_dict = self._dump_._get_opt_meta_data(self.memory_dict)
        return pd.DataFrame(results_dict["_score_"], columns=["_score_"])

    """
    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _obj2hash(self):
        obj2hash_dict = {}
        para_hash_list = self._get_para_hash_list()

        for para_hash in para_hash_list:
            obj = self._read_dill(para_hash)
            obj2hash_dict[para_hash] = obj

        return obj2hash_dict
    """
