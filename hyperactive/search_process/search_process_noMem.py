# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..candidate import CandidateNoMem
from .search_process_base import SearchProcess


class SearchProcessNoMem(SearchProcess):
    def __init__(self, nth_process, pro_arg, verb, hyperactive):
        super().__init__(nth_process, pro_arg, verb, hyperactive)

        self.cand = CandidateShortMem(
            self.obj_func,
            self.func_para,
            self.search_space,
            self.init_para,
            self.memory,
            verb,
            hyperactive,
        )
