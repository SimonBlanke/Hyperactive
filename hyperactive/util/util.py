# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np


def init_candidate(_core_, nth_process, Candidate):
    _core_._set_random_seed(nth_process)
    _cand_ = Candidate(nth_process, _core_)

    return _cand_


def init_eval(_cand_, nth_process, X, y):
    pos = _cand_._init_._set_start_pos(nth_process, X, y)
    score = _cand_.eval_pos(pos, X, y)
    _cand_.score_best = score
    _cand_.pos_best = pos

    return _cand_


def merge_dicts(base_dict, added_dict):
    # overwrite default values
    for key in base_dict.keys():
        if key in list(added_dict.keys()):
            base_dict[key] = added_dict[key]

    return base_dict


def sort_for_best(sort, sort_by):
    # Returns two lists sorted by the second
    sort = np.array(sort)
    sort_by = np.array(sort_by)

    index_best = list(sort_by.argsort()[::-1])

    sort_sorted = sort[index_best]
    sort_by_sorted = sort_by[index_best]

    return sort_sorted, sort_by_sorted
