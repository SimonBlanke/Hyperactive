# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from ..candidate import CandidateShortMem
from .search_process_shortMem import SearchProcessShortMem


class SearchProcessLongMem(SearchProcessShortMem):
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

