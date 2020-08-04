# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from ..candidate import CandidateShortMem
from .search_process_base import SearchProcess
from ..results_manager import ResultsManager


class SearchProcessShortMem(SearchProcess):
    def __init__(
        self,
        nth_process,
        p_bar,
        model,
        search_space,
        search_name,
        n_iter,
        training_data,
        optimizer,
        n_jobs,
        init_para,
        memory,
        random_state,
        verbosity,
    ):
        super().__init__(
            nth_process,
            p_bar,
            model,
            search_space,
            search_name,
            n_iter,
            training_data,
            optimizer,
            n_jobs,
            init_para,
            memory,
            random_state,
            verbosity,
        )

        self.cand = CandidateShortMem(
            self.model, self.training_data, self.search_space, self.init_para, p_bar,
        )

        if not isinstance(search_name, str):
            search_name = str(nth_process)

        self.res = ResultsManager(
            search_name, model, search_space, training_data, verbosity
        )

    def _memory2dataframe(self, memory):
        positions = np.array(list(memory.keys()))
        scores_list = list(memory.values())

        positions_df = pd.DataFrame(positions, columns=list(self.search_space.keys()))
        scores_df = pd.DataFrame(scores_list)

        self.position_results = pd.concat([positions_df, scores_df], axis=1)

