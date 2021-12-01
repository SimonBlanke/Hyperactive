# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd


class Results:
    def __init__(self, results_list, opt_pros):
        self.results_list = results_list
        self.opt_pros = opt_pros

        self.objFunc2results = {}
        self.search_id2results = {}

    def _sort_results_objFunc(self, objective_function):
        best_score = -np.inf
        best_para = None
        search_data = None

        results_list = []

        for results_ in self.results_list:
            nth_process = results_["nth_process"]

            opt = self.opt_pros[nth_process]
            objective_function_ = opt.objective_function
            search_space_ = opt.s_space()
            params = list(search_space_.keys())

            if objective_function_ != objective_function:
                continue

            if results_["best_score"] > best_score:
                best_score = results_["best_score"]
                best_para = results_["best_para"]

            results_list.append(results_["search_data"])

        if len(results_list) > 0:
            search_data = pd.concat(results_list)

        self.objFunc2results[objective_function] = {
            "best_para": best_para,
            "best_score": best_score,
            "search_data": search_data,
            "params": params,
        }

    def _sort_results_search_id(self, search_id):
        for results_ in self.results_list:
            nth_process = results_["nth_process"]
            search_id_ = self.process_infos[nth_process]["search_id"]

            if search_id_ != search_id:
                continue

            best_score = results_["best_score"]
            best_para = results_["best_para"]
            search_data = results_["search_data"]

            self.search_id2results[search_id] = {
                "best_para": best_para,
                "best_score": best_score,
                "search_data": search_data,
            }

    def _get_result(self, id_, result_name):
        if isinstance(id_, str):
            if id_ not in self.search_id2results:
                self._sort_results_search_id(id_)

            return self.search_id2results[id_][result_name]

        else:
            if id_ not in self.objFunc2results:
                self._sort_results_objFunc(id_)

            search_data = self.objFunc2results[id_][result_name]

            return search_data

    def best_para(self, id_):
        best_para_ = self._get_result(id_, "best_para")

        if best_para_ is not None:
            return best_para_

        raise ValueError("objective function name not recognized")

    def best_score(self, id_):
        best_score_ = self._get_result(id_, "best_score")

        if best_score_ != -np.inf:
            return best_score_

        raise ValueError("objective function name not recognized")

    def _create_mult_idx_search_data(self, search_data, params):
        level_0 = []
        level_1 = []
        for para in params:
            level_0.append("search_space")
            level_1.append(para)
        columns = list(search_data.columns)
        results_cols = [x for x in columns if x not in params]

        for results_col in results_cols:
            level_0.append("search_data")
            level_1.append(results_col)

        arrays = [level_0, level_1]
        mult_idx_tup = list(zip(*arrays))

        index = pd.MultiIndex.from_tuples(mult_idx_tup, names=["type", "info"])
        return pd.DataFrame(search_data.values, columns=index)

    def search_data(self, id_):
        search_data = self._get_result(id_, "search_data")

        params = self.objFunc2results[id_]["params"]
        # search_data = self._create_mult_idx_search_data(search_data, params)

        if search_data is not None:
            return search_data

        raise ValueError("objective function name not recognized")
