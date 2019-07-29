# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm

from .candidate import MlCandidate
from .candidate import DlCandidate


def initialize_search(_config_, nth_process, X, y):
    _cand_ = _init_candidate(_config_, nth_process)
    _cand_ = _init_eval(_cand_, nth_process, X, y)
    p_bar = _init_progress_bar(_config_, _cand_)

    return _cand_, p_bar


def _init_candidate(_config_, nth_process):
    _config_._set_random_seed(nth_process)

    if _config_.model_type == "sklearn" or _config_.model_type == "xgboost":
        _cand_ = MlCandidate(nth_process, _config_)

    elif _config_.model_type == "keras":
        _cand_ = DlCandidate(nth_process, _config_)

    return _cand_


def _init_eval(_cand_, nth_process, X, y):
    pos = _cand_._init_._set_start_pos(nth_process, X, y)
    score = _cand_.eval_pos(pos, X, y)
    _cand_.score_best = score
    _cand_.pos_best = pos

    return _cand_


def _init_progress_bar(_config_, _cand_):
    # create progress bar
    return tqdm.tqdm(**_config_._tqdm_dict(_cand_))
