# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import glob
import json
import dill
import pickle
import datetime
import hashlib
import inspect

import numpy as np
import pandas as pd


class Memory:
    def __init__(self, _space_, _main_args_, _cand_):
        self._space_ = _space_
        self._main_args_ = _main_args_

        self.pos_best = None
        self.score_best = -np.inf

        self.memory_type = _main_args_.memory
        self.memory_dict = {}

        self.meta_data_found = False


class ShortTermMemory(Memory):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)


class LongTermMemory(Memory):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)

        self.score_col_name = "mean_test_score"

        current_path = os.path.realpath(__file__)
        meta_learn_path, _ = current_path.rsplit("/", 1)

        self.datetime = datetime.datetime.now().strftime("%d.%m.%Y - %H:%M:%S") + "/"
        func_str = self._get_func_str(_cand_.func_)
        self.func_path_ = self._get_hash(func_str.encode("utf-8")) + "/"

        self.meta_path = meta_learn_path + "/meta_data/"
        self.func_path = self.meta_path + self.func_path_
        self.date_path = self.meta_path + self.func_path_ + self.datetime

        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path, exist_ok=True)

    def load_memory(self, model_func):
        para, score = self._read_func_metadata(model_func)
        if para is None or score is None:
            return

        self._load_data_into_memory(para, score)

    def save_memory(self, _main_args_, _opt_args_, _cand_):

        path = self._get_file_path(_cand_.func_)
        meta_data = self._collect(_cand_)

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

        os.chdir(self.date_path)
        os.system("black search_config.py")
        os.getcwd()

        run_data = {
            "random_state": self._main_args_.random_state,
            "max_time": self._main_args_.random_state,
            "n_iter": self._main_args_.n_iter,
            "optimizer": self._main_args_.optimizer,
            "n_jobs": self._main_args_.n_jobs,
            "eval_time": np.array(_cand_.eval_time).sum(),
            "total_time": _cand_.total_time,
        }

        with open("run_data.json", "w") as f:
            json.dump(run_data, f, indent=4)

        """

        print("_opt_args_.kwargs_opt", _opt_args_.kwargs_opt)

        opt_para = pd.DataFrame.from_dict(_opt_args_.kwargs_opt, dtype=object)
        print("opt_para", opt_para)
        opt_para.to_csv(
            self.meta_data_path + self.func_path + self.datetime + "opt_para",
            index=False,
        )
        """

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

            print("Loading meta data successful")
            return para, score

        else:
            print("Warning: No meta data found for following function:", model_func)
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

                    print("para", para[key])

                    para_dill = dill.dumps(para[key])
                    para_hash = self._get_hash(para_dill)

                    with open(
                        self.date_path + str(para_hash) + ".pkl", "wb"
                    ) as pickle_file:
                        dill.dump(para_dill, pickle_file)

                    para[key] = para_hash

            print("para", para["kernel"], type(para["kernel"]))

            if score != 0:
                para_list.append(para)
                score_list.append(score)

        results_dict["params"] = para_list
        results_dict["mean_test_score"] = score_list

        return results_dict

    def _load_data_into_memory(self, paras, scores):
        print("\nparas", paras)

        for idx in range(paras.shape[0]):
            para = paras.iloc[[idx]]

            print("\npara", para)

            pos = self._space_.para2pos(paras.iloc[[idx]], self._get_pkl_hash)
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

    def _collect(self, _cand_):
        para_pd = self._get_para()
        metric_pd = self._get_score()

        eval_time = pd.DataFrame(_cand_.eval_time, columns=["eval_time"])
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

    def _get_func_data_names(self):
        subdirs = self._get_subdirs()

        path_list = []
        for subdir in subdirs:
            paths = glob.glob(subdir + "*.csv")
            path_list = path_list + paths

        return path_list

    def _get_pkl_hash(self, hash):
        subdirs = self._get_subdirs()

        path_list = []
        for subdir in subdirs:
            paths = glob.glob(subdir + hash + "*.pkl")
            path_list = path_list + paths

        return path_list

    def _get_file_path(self, model_func):
        feature_hash = self._get_hash(self._main_args_.X)
        label_hash = self._get_hash(self._main_args_.y)

        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path)

        return self.date_path + (feature_hash + "_" + label_hash + ".csv")
