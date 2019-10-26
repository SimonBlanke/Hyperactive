# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


def initialize_search(_core_, nth_process, X, y):
    _cand_ = _init_candidate(_core_, nth_process)
    _cand_ = _init_eval(_cand_, nth_process, X, y)

    return _core_, _cand_
