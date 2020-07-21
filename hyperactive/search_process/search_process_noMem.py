# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..candidate import CandidateNoMem
from .search_process_base import SearchProcess
from ..results_manager import ResultsManager


class SearchProcessNoMem(SearchProcess):
    def __init__(
        self,
        nth_process,
        p_bar,
        objective_function,
        search_space,
        search_name,
        n_iter,
        function_parameter,
        optimizer,
        n_jobs,
        init_para,
        memory,
        random_state,
    ):
        super().__init__(
            nth_process,
            p_bar,
            objective_function,
            search_space,
            search_name,
            n_iter,
            function_parameter,
            optimizer,
            n_jobs,
            init_para,
            memory,
            random_state,
        )

        self.cand = CandidateNoMem(
            self.objective_function,
            self.function_parameter,
            self.search_space,
            self.init_para,
            self.memory,
            p_bar,
        )

        if not isinstance(search_name, str):
            search_name = str(nth_process)

        self.res = ResultsManager(
            search_name, objective_function, search_space, function_parameter
        )
