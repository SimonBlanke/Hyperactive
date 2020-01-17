# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import json
import dill
import time
import datetime
import hashlib
import inspect

import numpy as np
import pandas as pd

def apply_tobytes(df):
    return df.values.tobytes()




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

        self.nth_process = _cand_.nth_process

        self.score_col_name = "mean_test_score"

        self.feature_hash = self._get_hash(_main_args_.X)
        self.label_hash = self._get_hash(_main_args_.y)

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.datetime = "run_data/" + datetime.datetime.now().strftime("%d.%m.%Y - %H:%M:%S")
        func_str = self._get_func_str(_cand_.func_)
        self.func_path_ = self._get_hash(func_str.encode("utf-8")) + "/"

        self.meta_path = meta_learn_path + "/meta_data/"
        self.func_path = self.meta_path + self.func_path_
        self.date_path = self.meta_path + self.func_path_ + self.datetime + "/"

        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path, exist_ok=True)

    def load_memory(self, _cand_, _verb_):
        para, score = self._read_func_metadata(_cand_.func_, _verb_)
        if para is None or score is None:
            return

        _verb_.load_samples(para)
        _cand_.eval_time = list(para["eval_time"])

        self._load_data_into_memory(para, score)
        self.n_dims = len(para.columns)

    def save_memory(self, _main_args_, _opt_args_, _cand_):
        path = self._get_file_path(_cand_.func_)
        meta_data = self._collect(_cand_)

        meta_data["run"] = self.datetime
        self._save_toCSV(meta_data, path)

        obj_func_path = self.func_path + "objective_function.py"
        if not os.path.exists(obj_func_path):
            file = open(obj_func_path, "w")
            file.write(self._get_func_str(_cand_.func_))
            file.close()

        search_config_path = self.date_path + "search_config.py"
        search_config_temp = dict(self._main_args_.search_config)

        for key in search_config_temp.keys():
            if isinstance(key, str):
                continue
            search_config_temp[key.__name__] = search_config_temp[key]
            del search_config_temp[key]

        search_config_str = "search_config = " + str(search_config_temp)

        if not os.path.exists(search_config_path):
            file = open(search_config_path, "w")
            file.write(search_config_str)
            file.close()

        """
        os.chdir(self.date_path)
        os.system("black search_config.py")
        os.getcwd()
        """

        run_data = {
            "random_state": self._main_args_.random_state,
            "max_time": self._main_args_.random_state,
            "n_iter": self._main_args_.n_iter,
            "optimizer": self._main_args_.optimizer,
            "n_jobs": self._main_args_.n_jobs,
            "eval_time": np.array(_cand_.eval_time).sum(),
            "total_time": _cand_.total_time,
        }

        with open(self.date_path + "run_data.json", "w") as f:
            json.dump(run_data, f, indent=4)

        """
        print("_opt_args_.kwargs_opt", _opt_args_.kwargs_opt)

        opt_para = pd.DataFrame.from_dict(_opt_args_.kwargs_opt, dtype=object)
        print("opt_para", opt_para)
        opt_para.to_csv(self.date_path + "opt_para", index=False)
        """

    def _save_toCSV(self, meta_data_new, path):
        if os.path.exists(path):
            meta_data_old = pd.read_csv(path)

            if len(meta_data_old.columns) != len(meta_data_new.columns):
                print("Warning meta data dimensionality does not match")
                print("Meta data will not be saved")
                return

            meta_data = meta_data_old.append(meta_data_new)

            columns = list(meta_data.columns)
            noScore = ["mean_test_score", "cv_default_score", "eval_time", "run"]
            columns_noScore = [c for c in columns if c not in noScore]

            meta_data = meta_data.drop_duplicates(subset=columns_noScore)
        else:
            meta_data = meta_data_new

        meta_data.to_csv(path, index=False)

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

    def _get_opt_meta_data(self):
        results_dict = {}
        para_list = []
        score_list = []

        for key in self.memory_dict.keys():
            pos = np.fromstring(key, dtype=int)
            para = self._space_.pos2para(pos)
            score = self.memory_dict[key]

            for key in para.keys():
                if (
                    not isinstance(para[key], int)
                    and not isinstance(para[key], float)
                    and not isinstance(para[key], str)
                ):

                    para_dill = dill.dumps(para[key])
                    para_hash = self._get_hash(para_dill)

                    with open(
                        self.func_path + str(para_hash) + ".pkl", "wb"
                    ) as pickle_file:
                        dill.dump(para_dill, pickle_file)

                    para[key] = para_hash

            if score != 0:
                para_list.append(para)
                score_list.append(score)

        results_dict["params"] = para_list
        results_dict["mean_test_score"] = score_list

        return results_dict

    def _load_data_into_memory(self, paras, scores):

        paras = paras.replace(self._hash2obj())
        pos = self.para2pos(paras)

        if len(pos) == 0:
            return 

        df_temp = pd.DataFrame()
        df_temp["pos_str"] = pos.apply(apply_tobytes, axis=1)
        df_temp["score"] = scores

        self.memory_dict = df_temp.set_index('pos_str').to_dict()['score']

        scores = np.array(scores)
        paras = np.array(paras)

        idx = np.argmax(scores)
        self.score_best = scores[idx]
        self.pos_best = paras[idx]

    def apply_index(self, pos_key, df):
        return self._space_.search_space[pos_key].index(df) if df in self._space_.search_space[pos_key] else None

    def para2pos(self, paras):
        from functools import partial

        paras = paras[self._space_.para_names]
        pos = paras.copy()

        for pos_key in self._space_.search_space:
            apply_index = partial(self.apply_index, pos_key)
            pos[pos_key] = paras[pos_key].apply(
                apply_index
            )

        pos.dropna(how='any', inplace=True) 
        pos = pos.astype('int64')

        return pos

    def _collect(self, _cand_):
        results_dict = self._get_opt_meta_data()

        para_pd = pd.DataFrame(results_dict["params"])
        metric_pd = pd.DataFrame(
            results_dict["mean_test_score"], columns=["mean_test_score"]
        )

        eval_time = pd.DataFrame(_cand_.eval_time[-len(para_pd):], columns=["eval_time"])
        md_model = pd.concat(
            [para_pd, metric_pd, eval_time], axis=1, ignore_index=False
        )

        return md_model

    def _get_hash(self, object):
        return hashlib.sha1(object).hexdigest()

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_subdirs(self):
        subdirs = glob.glob(self.func_path + "*/")

        return subdirs

    def _get_func_data_names1(self):
        subdirs = self._get_subdirs()

        path_list = []
        for subdir in subdirs:
            paths = glob.glob(subdir + "*.csv")
            path_list = path_list + paths

        return path_list

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

    def _get_pkl_hash(self, hash):
        paths = glob.glob(self.func_path + hash + "*.pkl")

        return paths

    def _get_file_path(self, model_func):
        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path)

        return self.func_path + (self.feature_hash + "_" + self.label_hash + "_.csv")
