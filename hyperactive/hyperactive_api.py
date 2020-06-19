# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .main_args import MainArgs
from .general_optimizer import Optimizer


class Hyperactive:
    def __init__(
        self,
        X,
        y,
        max_time=None,
        memory="long",
        random_state=False,
        verbosity=3,
        warnings=False,
        ext_warnings=False,
    ):

        self._main_args_ = MainArgs(
            X, y, memory, random_state, verbosity, warnings, ext_warnings
        )

    def add_search(
        model,
        search_space,
        optimizer="RandomSearch",
        n_iter=10,
        n_jobs=1,
        init=None,
        distribution=None,
    ):
        pass

    def run():
        pass

    def search(
        self,
        search_config,
        n_iter=10,
        max_time=None,
        optimizer="RandomSearch",
        n_jobs=1,
        scheduler=None,
        init_config=None,
    ):
        self._main_args_.search_args(
            search_config, max_time, n_iter, optimizer, n_jobs, scheduler, init_config
        )

        core = HyperactiveCore(self._main_args_)
        core.run()

        self.results = core.results
        self.pos_list = core.pos_list
        # self.para_list = None
        self.score_list = core.score_list

        self.eval_times = core.eval_times
        self.iter_times = core.iter_times
        self.best_scores = core.best_scores
