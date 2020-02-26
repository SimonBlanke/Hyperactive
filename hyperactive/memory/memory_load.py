# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import json
import glob

import numpy as np
import pandas as pd

from functools import partial

from .memory_io import MemoryIO
from .util import get_model_id
from .paths import get_model_path


def apply_tobytes(df):
    return df.values.tobytes()


class MemoryLoad(MemoryIO):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.meta_data_found = False

        self.con_ids = []

        if not os.path.exists(self.meta_path + "model_connections.json"):
            with open(self.meta_path + "model_connections.json", "w") as f:
                json.dump({}, f, indent=4)

        with open(self.meta_path + "model_connections.json") as f:
            self.model_con = json.load(f)

        model_id = get_model_id(_cand_.func_)
        if model_id in self.model_con:
            self._get_id_list(self.model_con[model_id])
        else:
            self.con_ids = [model_id]

        self.con_ids = set(self.con_ids)

    def _get_id_list(self, id_list):
        self.con_ids = self.con_ids + id_list

        for id in id_list:
            id_list_new = self.model_con[id]

            if set(id_list_new).issubset(self.con_ids):
                continue

            self._get_id_list(id_list_new)

    def _load_memory(self, _cand_, _verb_, memory_dict):
        self.memory_dict = memory_dict

        para, score = self._read_func_metadata(_cand_.func_, _verb_)
        if para is None or score is None:
            return {}

        _verb_.load_samples(para)
        _cand_.eval_time = list(para["eval_time"])

        self._load_data_into_memory(para, score)
        self.n_dims = len(para.columns)

        return self.memory_dict

    def apply_index(self, pos_key, df):
        return (
            self._space_.search_space[pos_key].index(df)
            if df in self._space_.search_space[pos_key]
            else None
        )

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
        paths = []
        for id in self.con_ids:
            paths = paths + glob.glob(
                self.meta_path + get_model_path(id) + self.meta_data_name
            )

        return paths

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
        paras = paras.replace(self.hash2obj)
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
