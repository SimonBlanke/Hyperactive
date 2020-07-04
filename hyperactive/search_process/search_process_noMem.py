# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..candidate import CandidateNoMem
from .search_process_base import SearchProcess


class SearchProcessNoMem(SearchProcess):
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

        self.cand = CandidateNoMem(
            self.objective_function,
            self.function_parameter,
            self.search_space,
            self.init_para,
            self.memory,
            verb,
            hyperactive,
        )
