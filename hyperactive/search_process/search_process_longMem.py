# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from ..candidate import CandidateShortMem
from .search_process_shortMem import SearchProcessShortMem

from ..results_manager import ResultsManagerMemory

from optimization_metadata import HyperactiveWrapper
from ..meta_data.meta_data_path import meta_data_path


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

        if not isinstance(search_name, str):
            search_name = str(nth_process)

        self.hypermem = HyperactiveWrapper(
            main_path=meta_data_path(),
            X=training_data["features"],
            y=training_data["target"],
            model=model,
            search_space=search_space,
            verbosity=verbosity,
        )

        self.cand = CandidateShortMem(
            self.model, self.training_data, self.search_space, self.init_para, p_bar,
        )

        self.cand.memory_dict = self.load_long_term_memory()

    def load_long_term_memory(self):
        return self.hypermem.load()

    def save_long_term_memory(self):
        self.hypermem.save(self.memory_dict_new)

