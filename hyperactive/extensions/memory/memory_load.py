# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import dill
import hashlib
import inspect

import numpy as np
import pandas as pd

from functools import partial


def apply_tobytes(df):
    return df.values.tobytes()


class MemoryLoad:
    def __init__(self, _space_, _main_args_, _cand_, memory_dict):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = memory_dict

        self.score_col_name = "mean_test_score"

        self.meta_data_found = False

        self.feature_hash = self._get_hash(_main_args_.X)
        self.label_hash = self._get_hash(_main_args_.y)

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        func_str = self._get_func_str(_cand_.func_)
        self.func_path_ = self._get_hash(func_str.encode("utf-8")) + "/"

        self.meta_path = meta_learn_path + "/meta_data/"
        self.func_path = self.meta_path + self.func_path_

    def _load_memory(self, _cand_, _verb_):
        para, score = self._read_func_metadata(_cand_.func_, _verb_)
        if para is None or score is None:
            return

        _verb_.load_samples(para)
        _cand_.eval_time = list(para["eval_time"])

        self._load_data_into_memory(para, score)
        self.n_dims = len(para.columns)

    def apply_index(self, pos_key, df):
        return (
            self._space_.search_space[pos_key].index(df)
            if df in self._space_.search_space[pos_key]
            else None
        )

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _read_func_metadata(self, model_func, _verb_):
        paths = self._get_func_data_names()

        meta_data_list = []
        for path in paths:
            meta_data = pd.read_csv(path)
            meta_data_list.append(meta_data)
            self.meta_data_found = True

        if len(meta_data_list) > 0:
            meta_data = pd.concat(meta_data_list, ignore_index=True)

            column_names = meta_data.columns
            score_name = [name for name in column_names if self.score_col_name in name]

            para = meta_data.drop(score_name, axis=1)
            score = meta_data[score_name]

            _verb_.load_meta_data()
            return para, score

        else:
            _verb_.no_meta_data(model_func)
            return None, None

    def _get_func_data_names(self):
        paths = glob.glob(
            self.func_path + (self.feature_hash + "_" + self.label_hash + "_.csv")
        )

        return paths

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

    def _hash2obj(self):
        hash2obj_dict = {}
        para_hash_list = self._get_para_hash_list()

        for para_hash in para_hash_list:
            obj = self._read_dill(para_hash)
            hash2obj_dict[para_hash] = obj

        return hash2obj_dict

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def para2pos(self, paras):
        paras = paras[self._space_.para_names]
        pos = paras.copy()

        for pos_key in self._space_.search_space:
            apply_index = partial(self.apply_index, pos_key)
            pos[pos_key] = paras[pos_key].apply(apply_index)

        pos.dropna(how="any", inplace=True)
        pos = pos.astype("int64")

        return pos

    def _load_data_into_memory(self, paras, scores):

        paras = paras.replace(self._hash2obj())
        pos = self.para2pos(paras)

        if len(pos) == 0:
            return

        df_temp = pd.DataFrame()
        df_temp["pos_str"] = pos.apply(apply_tobytes, axis=1)
        df_temp["score"] = scores

        self.memory_dict = df_temp.set_index("pos_str").to_dict()["score"]

        scores = np.array(scores)
        paras = np.array(paras)

        idx = np.argmax(scores)
        self.score_best = scores[idx]
        self.pos_best = paras[idx]
