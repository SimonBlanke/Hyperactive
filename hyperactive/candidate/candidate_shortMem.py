# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .candidate_base import Candidate


class CandidateShortMem(Candidate):
    def __init__(self, obj_func, func_para, search_space, init_para, p_bar):
        super().__init__(obj_func, func_para, search_space, init_para, p_bar)

    def evaluate(self, pos, force_eval=False):
        pos.astype(int)
        pos_tuple = tuple(pos)

        if pos_tuple in self.memory_dict and not force_eval:
            return self.memory_dict[pos_tuple]["score"]
        else:
            results = self.base_eval(pos)
            self.memory_dict[pos_tuple] = results
            self.memory_dict_new[pos_tuple] = results

            return results["score"]

