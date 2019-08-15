# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import numpy as np
import pandas as pd

from importlib import import_module

from sklearn.model_selection import GridSearchCV
from .label_encoder import label_encoder_dict
from ..insight import Insight

from .data_wrangler import merge_meta_data, merge_dict, get_default_hyperpara


class Collector:
    def __init__(
        self, search_config, metric="accuracy", cv=5, n_jobs=1, meta_data_path=None
    ):
        self.search_config = search_config
        self.metric = metric
        self.cv = cv
        self.n_jobs = n_jobs
        self.meta_data_path = meta_data_path

        self.collector_model = ModelMetaDataCollector(self.search_config)

    def extract(self, X, y, _cand_list):
        self.insight = Insight(X, y)

        for model_str in self.search_config.keys():

            meta_data = self._get_meta_data(model_str, [X, y], _cand_list)
            self._save_toCSV(meta_data, model_str, self.insight.recognize_data())

    def _get_meta_data(self, model_name, data_train, _cand_list):
        X = data_train[0]
        y = data_train[1]

        md_model = self.collector_model.collect(model_name, X, y, _cand_list)
        md_dataset = self.insight.collect(model_name, data_train)

        meta_data = merge_meta_data(md_dataset, md_model)

        return meta_data

    def _save_toCSV(self, meta_data_new, model_str, data_hash):
        if not os.path.exists(self.meta_data_path):
            os.makedirs(self.meta_data_path)

        file_name = model_str + "___" + data_hash + "___metadata.csv"
        path = self.meta_data_path + file_name

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


class ModelMetaDataCollector:
    def __init__(self, search_config, cv=2, n_jobs=-1):
        self.search_config = search_config
        self.cv = cv
        self.n_jobs = n_jobs

        self.model_name = None
        self.hyperpara_dict = None

    def _get_opt_meta_data(self, _cand_list, model_name, X, y):
        results_dict = {}
        para_list = []
        score_list = []

        for _cand_ in _cand_list:
            for key in _cand_._space_.memory.keys():
                pos = np.fromstring(key)
                para = _cand_._space_.pos2para(pos)
                score = _cand_._space_.memory[key]

                if score != 0:
                    para_list.append(para)
                    score_list.append(score)

        results_dict["params"] = para_list
        results_dict["mean_test_score"] = score_list

        return results_dict

    def _get_grid_results(self, X, y, parameters):

        model_grid_search = GridSearchCV(
            self.model, parameters, cv=self.cv, n_jobs=self.n_jobs, verbose=1
        )
        model_grid_search.fit(X, y)

        grid_results = model_grid_search.cv_results_

        return grid_results

    def collect(self, model_name, X, y, _cand_list):
        self.hyperpara_dict = self._get_hyperpara(model_name)
        parameters = self.search_config[model_name]

        model = self._import_model(model_name)
        self.model = model()

        if _cand_list:
            results_dict = self._get_opt_meta_data(_cand_list, model_name, X, y)
        else:
            results_dict = self._get_grid_results(X, y, parameters)

        para_pd = pd.DataFrame(results_dict["params"])

        def_para_pd = get_default_hyperpara(self.model, len(para_pd))
        para_pd = merge_dict(para_pd, def_para_pd)

        para_pd = para_pd.reindex(sorted(para_pd.columns), axis=1)

        metric_pd = pd.DataFrame(
            results_dict["mean_test_score"], columns=["mean_test_score"]
        )

        md_model = pd.concat([para_pd, metric_pd], axis=1, ignore_index=False)
        md_model = self._label_enc(md_model) * 1  # to convert False to 0 and True to 1

        return md_model

    def _get_hyperpara(self, model_name):
        return label_encoder_dict[model_name]

    def _label_enc(self, X_train):
        for hyperpara_key in self.hyperpara_dict:
            to_replace = {hyperpara_key: self.hyperpara_dict[hyperpara_key]}
            X_train = X_train.replace(to_replace)
        X_train = X_train.infer_objects()

        return X_train

    def _import_model(self, model):
        sklearn, submod_func = model.rsplit(".", 1)
        module = import_module(sklearn)
        model = getattr(module, submod_func)

        return model
