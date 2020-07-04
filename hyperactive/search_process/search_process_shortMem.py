# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from ..candidate import CandidateShortMem
from .search_process_base import SearchProcess


class SearchProcessShortMem(SearchProcess):
    def __init__(
        self,
        nth_process,
        verb,
        objective_function,
        search_space,
        n_iter,
        function_parameter,
        optimizer,
        n_jobs,
        init_para,
        memory,
        hyperactive,
        random_state,
    ):
        super().__init__(
            nth_process,
            verb,
            objective_function,
            search_space,
            n_iter,
            function_parameter,
            optimizer,
            n_jobs,
            init_para,
            memory,
            hyperactive,
            random_state,
        )

        self.cand = CandidateShortMem(
            self.objective_function,
            self.function_parameter,
            self.search_space,
            self.init_para,
            self.memory,
            verb,
            hyperactive,
        )

    def _memory2dataframe(self, memory):
        positions = np.array(list(memory.keys()))
        scores_list = list(memory.values())

        positions_df = pd.DataFrame(positions, columns=list(self.search_space.keys()))
        scores_df = pd.DataFrame(scores_list)

        self.position_results = pd.concat([positions_df, scores_df], axis=1)

