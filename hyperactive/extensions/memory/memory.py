# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import json
import shutil
import hashlib
import inspect

import numpy as np

from .memory_load import MemoryLoad
from .memory_dump import MemoryDump


class Memory:
    def __init__(self):
        current_path = os.path.realpath(__file__)
        self.meta_learn_path, _ = current_path.rsplit("/", 1)
        self.meta_path = self.meta_learn_path + "/meta_data/"

    def get_best_models(self, X, y):
        # TODO: model_dict   key:model   value:score

        return model_dict

    def get_model_search_config(self, model):
        # TODO
        return search_config

    def get_model_init_config(self, model):
        # TODO
        return init_config

    def delete_model(self, model):
        model_hash = self._get_model_hash(model)
        shutil.rmtree(self.meta_path + str(model_hash))

    def delete_model_dataset(self, model, X, y):
        self.func_path_ = self._get_model_hash(model) + "/"
        self.func_path = self.meta_path + self.func_path_

        self.feature_hash = self._get_hash(X)
        self.label_hash = self._get_hash(y)

        csv_file = self._get_file_path()
        os.remove(csv_file)

    def merge_model_hashes(self, model1, model2):
        # do checks if search space has same dim

        with open(self.meta_path + 'model_connections.json') as f:
            data = json.load(f)

        model1_hash = self._get_model_hash(model1)
        model2_hash = self._get_model_hash(model2)

        models_dict = {str(model1_hash): str(model2_hash)}
        data.update(models_dict)

        with open(self.meta_path + 'model_connections.json', 'w') as f:
            json.dump(data, f)

    def split_model_hashes(self, model1, model2):
        # TODO: do checks if search space has same dim

        with open(self.meta_path + 'model_connections.json') as f:
            data = json.load(f)

        model1_hash = self._get_model_hash(model1)
        model2_hash = self._get_model_hash(model2)

        if model1_hash in data.keys():
            del data[model1_hash]
        if model2_hash in data.keys():
            del data[model2_hash]

        with open(self.meta_path + 'model_connections.json', 'w') as f:
            json.dump(data, f)

    def _get_model_hash(self, model):
        return self._get_hash(self._get_func_str(model).encode("utf-8"))

    def _get_file_path(self):
        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path)

        return self.func_path + (self.feature_hash + "_" + self.label_hash + "_.csv")

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()


class BaseMemory:
    def __init__(self, _space_, _main_args_, _cand_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = {}

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
        self._dump_._save_memory(_main_args_, _opt_args_, _cand_, self.memory_dict)

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
