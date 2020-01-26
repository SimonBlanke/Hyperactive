# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import json
import dill
import datetime
import hashlib
import inspect

import numpy as np
import pandas as pd

from functools import partial

from .memory_load import MemoryLoad
from .memory_dump import MemoryDump


class Memory:
    def __init__(self, _space_, _main_args_, _cand_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = {}

        self.meta_data_found = False

        self.n_dims = None


class ShortTermMemory(Memory):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)


class LongTermMemory(Memory):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)

        self._load_ = MemoryLoad(_space_, _main_args_, _cand_, self.memory_dict)
        self._dump_ = MemoryDump(_space_, _main_args_, _cand_, self.memory_dict)

        self.nth_process = _cand_.nth_process

        self.score_col_name = "mean_test_score"

        self.feature_hash = self._get_hash(_main_args_.X)
        self.label_hash = self._get_hash(_main_args_.y)

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.datetime = "run_data/" + datetime.datetime.now().strftime(
            "%d.%m.%Y - %H:%M:%S"
        )
        func_str = self._get_func_str(_cand_.func_)
        self.func_path_ = self._get_hash(func_str.encode("utf-8")) + "/"

        self.meta_path = meta_learn_path + "/meta_data/"
        self.func_path = self.meta_path + self.func_path_
        self.date_path = self.meta_path + self.func_path_ + self.datetime + "/"

        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path, exist_ok=True)

    def load_memory(self, _cand_, _verb_):
        self._load_._load_memory(_cand_, _verb_)

    def save_memory(self, _main_args_, _opt_args_, _cand_):
        self._dump_._save_memory(_main_args_, _opt_args_, _cand_)

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _obj2hash(self):
        hash2obj_dict = {}
        para_hash_list = self._get_para_hash_list()

        for para_hash in para_hash_list:
            obj = self._read_dill(para_hash)
            hash2obj_dict[para_hash] = obj

        return obj2hash_dict
