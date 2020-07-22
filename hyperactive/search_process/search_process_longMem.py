# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from ..candidate import CandidateShortMem
from .search_process_shortMem import SearchProcessShortMem

from ..results_manager import ResultsManagerMemory


class SearchProcessLongMem(SearchProcessShortMem):
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
        )

        if not isinstance(search_name, str):
            search_name = str(nth_process)

        self.res = ResultsManagerMemory(search_name, model, search_space, training_data)

        self.cand = CandidateShortMem(
            self.model,
            self.training_data,
            self.search_space,
            self.init_para,
            self.memory,
            p_bar,
        )

        self.cand.memory_dict = self.res.load_long_term_memory()

