# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from .distribution import single_process, joblib_wrapper, multiprocessing_wrapper
from .model_info import print_results_info


class SearchManager:
    def __init__(self, search_info):
        self.s_info = search_info

        # self.n_processes = len(search_processes)

        self.results = {}
        self.eval_times = {}
        self.iter_times = {}
        self.best_scores = {}
        self.pos_list = {}
        self.score_list = {}
        self.position_results = {}

    """

    def _get_results(self, results_list):
        position_results_dict = {}

        self.eval_times_dict = {}
        self.iter_times_dict = {}

        self.positions_dict = {}
        self.scores_dict = {}
        self.best_score_list_dict = {}

        self.para_best_dict = {}
        self.score_best_dict = {}
        self.memory_dict_new = {}

        

        search_name_n_jobs = {}
        for results in results_list:
            search_name_n_jobs[results.search_name] = results.n_jobs

        for results in results_list:
            search_name = results.search_name
            search_id = str(results.model.__name__) + "." + str(results.nth_process)

            self.eval_times_dict[search_id] = results.eval_times
            self.iter_times_dict[search_id] = results.iter_times

            self.positions_dict[search_id] = results.pos_list
            self.scores_dict[search_id] = results.score_list
            self.best_score_list_dict[search_id] = results.best_score_list

            self.para_best_dict[search_id] = results.para_best
            self.score_best_dict[search_id] = results.score_best
            self.memory_dict_new[search_id] = results.memory_dict_new
            self.position_results[search_id] = self._memory_dict2dataframe(
                results.memory_dict_new, results.search_space
            )

            if self.verbosity == 0:
                continue

            print("\nSearch-ID:", search_id)
            print("  best parameter =", results.para_best)
            print("  best score     =", results.score_best)

            if results.memory == "long":
                results.save_long_term_memory()





        self.hypermem = HyperactiveWrapper(
            main_path=meta_data_path(),
            X=training_data["features"],
            y=training_data["target"],
            model=model,
            search_space=search_space,
            verbosity=verbosity,
        )

        self.cand = CandidateShortMem(
            self.model, self.training_data, self.search_space, self.init_para,
        )

        self.cand.memory_dict = self.load_long_term_memory()

    def load_long_term_memory(self):
        return self.hypermem.load()

    def save_long_term_memory(self):
        self.hypermem.save(self.memory_dict_new)

        

    def _memory_dict2dataframe(self, memory_dict, search_space):
        columns = list(search_space.keys())

        if not bool(memory_dict):
            return pd.DataFrame([], columns=columns)

        pos_tuple_list = list(memory_dict.keys())
        result_list = list(memory_dict.values())

        results_df = pd.DataFrame(result_list)
        np_pos = np.array(pos_tuple_list)

        pd_pos = pd.DataFrame(np_pos, columns=columns)
        dataframe = pd.concat([pd_pos, results_df], axis=1)

        return dataframe
    """

    def run(self, process_info_dict):
        self.s_info.print_search_info()
        process_infos = list(process_info_dict.values())

        if len(process_infos) == 1:
            results_list = single_process(process_infos)
        else:
            results_list = multiprocessing_wrapper(process_infos)

        self.results_list = results_list

        print_results_info(results_list, process_info_dict)

