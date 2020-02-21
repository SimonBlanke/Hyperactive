# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import json
import dill
import inspect

import numpy as np
import pandas as pd

from .memory_io import MemoryIO
from .dataset_features import get_dataset_features
from .util import get_hash


class MemoryDump(MemoryIO):
    def __init__(self, _space_, _main_args_, _cand_):
        super().__init__(_space_, _main_args_, _cand_)

        self.memory_type = _main_args_.memory

    def _save_memory(self, _main_args_, _opt_args_, _cand_, memory_dict):
        self.memory_dict = memory_dict

        # Save meta_data
        path = self._get_file_path(_cand_.func_)
        meta_data = self._collect(_cand_, memory_dict)

        if meta_data is None:
            return

        meta_data["run"] = self.datetime
        self._save_toCSV(meta_data, path)

        # Save function
        obj_func_path = self.model_path + "objective_function.pkl"

        with open(obj_func_path, "wb") as pickle_file:
            dill.dump(_cand_.func_, pickle_file)

        # Save search space
        search_space_path = self.model_path + "search_space.pkl"

        with open(search_space_path, "wb") as pickle_file:
            dill.dump(_cand_.search_space, pickle_file)

        """
        if not os.path.exists(obj_func_path):
            file = open(obj_func_path, "w")
            file.write(self._get_func_str(_cand_.func_))
            file.close()
        """
        # Save search_config
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

        # Save data_features
        data_features = get_dataset_features(_main_args_.X, _main_args_.y)

        if not os.path.exists(self.dataset_info_path):
            os.makedirs(self.dataset_info_path, exist_ok=True)

        with open(self.dataset_info_path + "data_features.json", "w") as f:
            json.dump(data_features, f, indent=4)

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

    def _get_func_str(self, func):
        return inspect.getsource(func)

    def _get_file_path(self, model_func):
        if not os.path.exists(self.date_path):
            os.makedirs(self.date_path)

        return self.model_path + self.meta_data_name

    def _collect(self, _cand_, memory_dict):
        results_dict = self._get_opt_meta_data(memory_dict)

        if not results_dict:
            return None

        para_pd = pd.DataFrame(results_dict["params"])
        metric_pd = pd.DataFrame(results_dict["_score_"], columns=["_score_"])
        n_rows = len(para_pd)
        eval_time = pd.DataFrame(_cand_.eval_time[-n_rows:], columns=["eval_time"])
        md_model = pd.concat(
            [para_pd, metric_pd, eval_time], axis=1, ignore_index=False
        )

        return md_model

    def _get_opt_meta_data(self, memory_dict):
        results_dict = {}
        para_list = []
        score_list = []

        if not memory_dict:
            return None

        for key in memory_dict.keys():
            pos = np.fromstring(key, dtype=int)
            para = self._space_.pos2para(pos)
            score = memory_dict[key]

            for key in para.keys():
                if (
                    not isinstance(para[key], int)
                    and not isinstance(para[key], float)
                    and not isinstance(para[key], str)
                ):

                    para_dill = dill.dumps(para[key])
                    para_hash = get_hash(para_dill)

                    with open(
                        self.model_path + str(para_hash) + ".pkl", "wb"
                    ) as pickle_file:
                        dill.dump(para_dill, pickle_file)

                    para[key] = para_hash

            if score != 0:
                para_list.append(para)
                score_list.append(score)

        results_dict["params"] = para_list
        results_dict["_score_"] = score_list

        return results_dict

    def _save_toCSV(self, meta_data_new, path):
        if os.path.exists(path):
            meta_data_old = pd.read_csv(path)

            assert len(meta_data_old.columns) == len(
                meta_data_new.columns
            ), "Warning meta data dimensionality does not match"

            meta_data = meta_data_old.append(meta_data_new)

            columns = list(meta_data.columns)
            noScore = ["_score_", "cv_default_score", "eval_time", "run"]
            columns_noScore = [c for c in columns if c not in noScore]

            meta_data = meta_data.drop_duplicates(subset=columns_noScore)
        else:
            meta_data = meta_data_new

        meta_data.to_csv(path, index=False)
