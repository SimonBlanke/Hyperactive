# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .candidate_base import Candidate


class CandidateNoMem(Candidate):
    def __init__(self, obj_func, func_para, search_space, init_para, p_bar):
        super().__init__(obj_func, func_para, search_space, init_para, p_bar)

    def evaluate(self, pos, nth_iter):
        pos.astype(int)
        pos_tuple = tuple(pos)

        results = self.base_eval(pos, nth_iter)
        if pos_tuple not in self.memory_dict_new:
            self.memory_dict_new[pos_tuple] = results

        return results["score"]

