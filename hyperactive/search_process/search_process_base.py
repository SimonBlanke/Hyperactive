# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import time
import random
import numpy as np
import pandas as pd


from ..io_search_processor import IoSearchProcessor


class SearchProcess:
    def __init__(
        self,
        nth_process,
        p_bar,
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
        self.p_bar = p_bar
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
            p_bar,
            model,
            search_space,
            n_iter,
            optimizer,
            init_para,
            random_state,
        )

    def _time_exceeded(self, start_time, max_time):
        run_time = time.time() - start_time
        return max_time and run_time > max_time

    def _save_results(self):
        # self.res.nth_process = self.nth_process
        self.eval_times = self.eval_times
        self.iter_times = self.iter_times

        self.pos_list = self.cand.pos_list
        self.score_list = self.cand.score_list
        self.best_score_list = self.cand.scores_best_list

        # self.res.n_jobs = self.n_jobs
        self.memory_dict_new = self.cand.memory_dict_new
        self.para_best = self.cand.para_best
        self.score_best = self.cand.score_best
        # self.res.model = self.model
        # self.res.search_space = self.search_space
        # self.res.memory = self.memory

    def search(self, start_time, max_time, nth_process):
        start_time_search = time.time()
        self.opt = self.search_io.init_search(nth_process, self.cand)

        # loop to initialize N positions
        for nth_init in range(len(self.opt.init_positions)):
            start_time_iter = time.time()
            pos_new = self.opt.init_pos(nth_init)

            start_time_eval = time.time()
            score_new = self.cand.get_score(pos_new)
            self.eval_times.append(time.time() - start_time_eval)

            self.opt.evaluate(score_new)
            self.iter_times.append(time.time() - start_time_iter)

        # loop to do the iterations
        for nth_iter in range(len(self.opt.init_positions), self.n_iter):
            start_time_iter = time.time()
            pos_new = self.opt.iterate(nth_iter)

            start_time_eval = time.time()
            score_new = self.cand.get_score(pos_new)
            self.eval_times.append(time.time() - start_time_eval)

            self.opt.evaluate(score_new)
            self.iter_times.append(time.time() - start_time_search)

            if self._time_exceeded(start_time, max_time):
                break

        self.p_bar.close_p_bar()
        self._save_results()

        return self

