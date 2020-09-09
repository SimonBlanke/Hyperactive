# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def pos2para(search_space, pos):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(pos[i])
        values_dict[key] = search_space[key][pos_]

    return values_dict


def _process_(
    nth_process,
    model,
    search_space,
    n_iter,
    name,
    opt,
    initialize,
    memory,
    max_time,
    distribution,
    X,
    y,
    random_state,
    verbosity,
):
    def gfo_wrapper_model():
        # rename _model
        def _model(array):
            # wrapper for GFOs
            para = pos2para(search_space, array)
            return model(para, X, y)

        _model.__name__ = model.__name__
        return _model

    verbosity["print_results"] = False

    opt.search(
        objective_function=gfo_wrapper_model(),
        n_iter=n_iter,
        initialize=initialize,
        max_time=max_time,
        memory=memory,
        verbosity=verbosity,
        random_state=random_state,
        nth_process=nth_process,
    )

    return {
        "model_name": model.__name__,
        "search_space": search_space,
        "best_pos": opt.best_values,
        "best_score": opt.best_score,
        "values": opt.values,
        "scores": opt.scores,
    }
