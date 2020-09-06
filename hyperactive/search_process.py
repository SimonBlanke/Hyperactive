# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


from .io_search_processor import IoSearchProcessor


def pos2para(search_space, pos):
    values_dict = {}
    for i, key in enumerate(search_space.keys()):
        pos_ = int(pos[i])
        values_dict[key] = search_space[key][pos_]

    return values_dict


class SearchProcess:
    def __init__(
        self,
        nth_process,
        model,
        search_space,
        search_name,
        n_iter,
        training_data,
        optimizer,
        n_jobs,
        init_para,
        memory,
        random_state,
        verbosity,
    ):
        self.nth_process = nth_process
        self.model = model
        self.search_space = search_space
        self.search_name = search_name
        self.n_iter = n_iter
        self.training_data = training_data
        self.optimizer = optimizer
        self.n_jobs = n_jobs
        self.init_para = init_para
        self.memory = memory
        self.random_state = random_state
        self.verbosity = verbosity

        self.iter_times = []
        self.eval_times = []

        self.search_io = IoSearchProcessor(
            nth_process,
            model,
            search_space,
            n_iter,
            optimizer,
            init_para,
            random_state,
        )

    def search(self, start_time, max_time, nth_process):
        start_time_search = time.time()

        self.search_space_pos = []

        for dict_value in self.search_space.values():
            space_dim = np.array(range(len(dict_value)))
            self.search_space_pos.append(space_dim)

        self.opt = self.search_io.init_search(nth_process, self.search_space_pos)

        X = self.training_data["features"]
        y = self.training_data["target"]

        def gfo_wrapper_model():
            # rename _model
            def _model(array):
                # wrapper for GFOs
                para = pos2para(self.search_space, array)
                return self.model(para, X, y)

            _model.__name__ = self.model.__name__
            return _model

        self.opt.search(
            objective_function=gfo_wrapper_model(),
            n_iter=self.n_iter,
            # initialize={"grid": 7, "random": 3,},
            max_time=max_time,
            memory=True,
            print_results=False,
            progress_bar=self.verbosity,
            random_state=self.random_state,
            nth_process=nth_process,
        )

        self.opt.p_bar._tqdm = tqdm(
            total=self.n_iter, leave=False, position=nth_process
        )

        return self.opt

