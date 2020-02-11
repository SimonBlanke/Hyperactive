# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import json
import glob
import hashlib

import numpy as np
import pandas as pd

from functools import partial

from .memory_io import MemoryIO


def apply_tobytes(df):
    return df.values.tobytes()


class MemoryLoad(MemoryIO):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.meta_data_found = False

        with open(self.meta_path + "model_connections.json") as f:
            model_connections = json.load(f)

        print("\nmodel_connections\n", model_connections)

        model_id = self._get_model_hash(_cand_.func_)
        self.connected_ids = [model_id]

        print("\nmodel_id\n", model_id)

        if model_id in model_connections:
            self.connected_ids = self.connected_ids + model_connections[model_id]
        self._get_id_list(model_connections, model_id)
        self.connected_ids = set(self.connected_ids)

        print("\nself.connected_ids\n", self.connected_ids)

    def _get_id_list(self, model_connections, id):
        if id in model_connections:
            id_list = model_connections[id]
        else:
            return

        for id in id_list:
            if id in model_connections:
                self.connected_ids = self.connected_ids + model_connections[id]

            self._get_id_list(model_connections, id)

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

        print("\npaths\n", paths)

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
        for id in self.connected_ids:
            paths = paths + glob.glob(
                self.meta_path
                + id
                + "/"
                + (self.feature_hash + "_" + self.label_hash + "_.csv")
            )

        return paths

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
