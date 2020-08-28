# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..candidate import CandidateNoMem
from .search_process_base import SearchProcess


class SearchProcessNoMem(SearchProcess):
    def __init__(
        self,
        nth_process,
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

        self.cand = CandidateNoMem(
            self.model, self.training_data, self.search_space, self.init_para,
        )

        if not isinstance(search_name, str):
            search_name = str(nth_process)

