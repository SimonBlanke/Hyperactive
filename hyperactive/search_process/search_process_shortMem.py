# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd

from ..candidate import CandidateShortMem
from .search_process_base import SearchProcess


class SearchProcessShortMem(SearchProcess):
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

    def _memory2dataframe(self, memory):
        positions = np.array(list(memory.keys()))
        scores_list = list(memory.values())

        positions_df = pd.DataFrame(positions, columns=list(self.search_space.keys()))
        scores_df = pd.DataFrame(scores_list)

        self.position_results = pd.concat([positions_df, scores_df], axis=1)

