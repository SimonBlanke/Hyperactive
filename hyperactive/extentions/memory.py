# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import hashlib
import inspect

import numpy as np
import pandas as pd


class Memory:
    def __init__(self, _space_, _main_args_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = {}

        self.meta_data_found = False

    def load_memory(self, model_func):
        pass

    def save_memory(self, _main_args_, _cand_):
        pass


class ShortTermMemory(Memory):
    def __init__(self, _space_, _main_args_):
        super().__init__(_space_, _main_args_)


class LongTermMemory(Memory):
    def __init__(self, _space_, _main_args_):
        super().__init__(_space_, _main_args_)

        self.score_col_name = "mean_test_score"

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)
        self.meta_data_path = meta_learn_path + "/meta_data/"

    def load_memory(self, model_func):
        para, score = self._read_func_metadata(model_func)
        if para is None or score is None:
            return

        self._load_data_into_memory(para, score)

    def save_memory(self, _main_args_, _cand_):
        meta_data = self._collect()
        path = self._get_file_path(_cand_.func_)
        self._save_toCSV(meta_data, path)

    def _save_toCSV(self, meta_data_new, path):
        if os.path.exists(path):
            meta_data_old = pd.read_csv(path)
            meta_data = meta_data_old.append(meta_data_new)

            columns = list(meta_data.columns)
            noScore = ["mean_test_score", "cv_default_score"]
            columns_noScore = [c for c in columns if c not in noScore]

            meta_data = meta_data.drop_duplicates(subset=columns_noScore)
        else:
            meta_data = meta_data_new

        meta_data.to_csv(path, index=False)

    def _read_func_metadata(self, model_func):
        paths = glob.glob(self._get_func_file_paths(model_func))

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

            return para, score

        else:
            print("Warning: No meta data found for following function:", model_func)
            return None, None

    def _get_opt_meta_data(self):
        results_dict = {}
        para_list = []
        score_list = []

        for key in self.memory_dict.keys():
            pos = np.fromstring(key, dtype=float)
            para = self._space_.pos2para(pos)
            score = self.memory_dict[key]

            if score != 0:
                para_list.append(para)
                score_list.append(score)

        results_dict["params"] = para_list
        results_dict["mean_test_score"] = score_list

        return results_dict

    def _load_data_into_memory(self, paras, scores):
        for idx in range(paras.shape[0]):
            pos = self._space_.para2pos(paras.iloc[[idx]])
            pos_str = pos.tostring()

            score = float(scores.values[idx])
            self.memory_dict[pos_str] = score

            if score > self.score_best:
                self.score_best = score
                self.pos_best = pos

    def _get_para(self):
        results_dict = self._get_opt_meta_data()

        return pd.DataFrame(results_dict["params"])

    def _get_score(self):
        results_dict = self._get_opt_meta_data()
        return pd.DataFrame(
            results_dict["mean_test_score"], columns=["mean_test_score"]
        )

    def _collect(self):
        para_pd = self._get_para()
        md_model = para_pd.reindex(sorted(para_pd.columns), axis=1)
        metric_pd = self._get_score()

        md_model = pd.concat([para_pd, metric_pd], axis=1, ignore_index=False)

        return md_model

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_func_file_paths(self, model_func):
        func_str = self._get_func_str(model_func)
        self.func_path = self._get_hash(func_str.encode("utf-8")) + "/"

        directory = self.meta_data_path + self.func_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory + ("metadata" + "*" + "__.csv")

    def _get_file_path(self, model_func):
        func_str = self._get_func_str(model_func)
        feature_hash = self._get_hash(self._main_args_.X)
        label_hash = self._get_hash(self._main_args_.y)

        self.func_path = self._get_hash(func_str.encode("utf-8")) + "/"

        directory = self.meta_data_path + self.func_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory + (
            "metadata"
            + "__feature_hash="
            + feature_hash
            + "__label_hash="
            + label_hash
            + "__.csv"
        )
