# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from ..candidate import Candidate


def initialize_search(_config_, nth_process, X, y):
    _cand_ = _init_candidate(_config_, nth_process)
    _cand_ = _init_eval(_cand_, nth_process, X, y)
    _config_.init_p_bar(_config_, _cand_)

    return _config_, _cand_


def _init_candidate(_config_, nth_process):
    _config_._set_random_seed(nth_process)
    _cand_ = Candidate(nth_process, _config_)

    return _cand_


def _init_eval(_cand_, nth_process, X, y):
    pos = _cand_._init_._set_start_pos(nth_process, X, y)
    score = _cand_.eval_pos(pos, X, y)
    _cand_.score_best = score
    _cand_.pos_best = pos

    return _cand_
