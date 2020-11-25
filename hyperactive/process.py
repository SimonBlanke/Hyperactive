# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def gfo2hyper(search_space, para):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(para[key])
        values_dict[key] = search_space[key][pos_]

    return values_dict


def _process_(
    nth_process,
    model,
    search_space,
    optimizer,
    n_iter,
    initialize,
    memory,
    max_time,
    random_state,
    verbosity,
):
    def gfo_wrapper_model():
        # wrapper for GFOs
        def _model(para):
            para = gfo2hyper(search_space, para)
            optimizer.suggested_params = para
            return model(optimizer)

        _model.__name__ = model.__name__
        return _model

    # verbosity["print_results"] = False

    optimizer.search(
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
        "nth_process": nth_process,
        "best_para": optimizer.best_para,
        "best_score": optimizer.best_score,
        "results": optimizer.results,
        "memory_dict_new": optimizer.memory_dict_new,
    }
