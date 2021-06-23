# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd


class HyperactiveResults:
    def __init__(self, *args, **kwargs):
        self.objFunc2results = {}
        self.search_id2results = {}

    def _sort_results_objFunc(self, objective_function):
        best_score = -np.inf
        best_para = None
        search_data = None

        results_list = []

        for results_ in self.results_list:
            nth_process = results_["nth_process"]

            process_infos = self.process_infos[nth_process]
            objective_function_ = process_infos["objective_function"]

            if objective_function_ != objective_function:
                continue

            if results_["best_score"] > best_score:
                best_score = results_["best_score"]
                best_para = results_["best_para"]

            results_list.append(results_["results"])

        if len(results_list) > 0:
            search_data = pd.concat(results_list)

        self.objFunc2results[objective_function] = {
            "best_para": best_para,
            "best_score": best_score,
            "search_data": search_data,
        }

    def _sort_results_search_id(self, search_id):
        for results_ in self.results_list:
            nth_process = results_["nth_process"]
            search_id_ = self.process_infos[nth_process]["search_id"]

            if search_id_ != search_id:
                continue

            best_score = results_["best_score"]
            best_para = results_["best_para"]
            search_data = results_["results"]

            self.search_id2results[search_id] = {
                "best_para": best_para,
                "best_score": best_score,
                "search_data": search_data,
            }

    def _get_one_result(self, id_, result_name):
        if isinstance(id_, str):
            if id_ not in self.search_id2results:
                self._sort_results_search_id(id_)

            return self.search_id2results[id_][result_name]

        else:
            if id_ not in self.objFunc2results:
                self._sort_results_objFunc(id_)

            return self.objFunc2results[id_][result_name]

    def best_para(self, id_):
        best_para_ = self._get_one_result(id_, "best_para")

        if best_para_ is not None:
            return best_para_

        raise ValueError("objective function name not recognized")

    def best_score(self, id_):
        best_score_ = self._get_one_result(id_, "best_score")

        if best_score_ != -np.inf:
            return best_score_

        raise ValueError("objective function name not recognized")

    def results(self, id_):
        results_ = self._get_one_result(id_, "search_data")

        if results_ is not None:
            return results_

        raise ValueError("objective function name not recognized")
