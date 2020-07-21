# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .candidate_base import Candidate


class CandidateShortMem(Candidate):
    def __init__(self, obj_func, func_para, search_space, init_para, memory, verb):
        super().__init__(obj_func, func_para, search_space, init_para, memory, verb)
