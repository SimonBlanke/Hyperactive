# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np


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
