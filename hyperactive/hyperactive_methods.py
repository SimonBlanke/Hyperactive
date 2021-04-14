# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd


class HyperactiveResults:
    def __init__(*args, **kwargs):
        pass

    def _sort_results_objFunc(self, id_):
        best_score = -np.inf
        best_para = None
        search_data = None

        results_list = []

        for results_ in self.results_list:
            nth_process = results_["nth_process"]

            process_infos = self.process_infos[nth_process]
            objective_function_ = process_infos["objective_function"]

            if objective_function_.__name__ != id_:
                continue

            if results_["best_score"] > best_score:
                best_score = results_["best_score"]
                best_para = results_["best_para"]

            results_list.append(results_["results"])

        if len(results_list) > 0:
            search_data = pd.concat(results_list)

        self.objFunc2results[id_] = {
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
            search_id = id_

            if search_id not in self.search_id2results:
                self._sort_results_search_id(search_id)

            return self.search_id2results[search_id][result_name]

        else:
            objective_function = id_

            if objective_function not in self.objFunc2results:
                self._sort_results_objFunc(objective_function.__name__)

            return self.objFunc2results[objective_function.__name__][result_name]

    def best_para(self, id_):
        return self._get_one_result(id_, "best_para")

    def best_score(self, id_):
        return self._get_one_result(id_, "best_score")

    def results(self, id_):
        return self._get_one_result(id_, "search_data")
