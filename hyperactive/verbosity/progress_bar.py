# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm.auto import tqdm


class ProgressBar:
    def init_p_bar(self, _cand_, _core_):
        pass

    def update_p_bar(self, n, _cand_):
        pass

    def close_p_bar(self):
        pass


class ProgressBarLVL0(ProgressBar):
    def __init__(self):
        self.best_since_iter = 0

    def init_p_bar(self, nth_process, _core_):
        self.p_bar = tqdm(**self._tqdm_dict(nth_process, _core_))

    def update_p_bar(self, n, _cand_):
        self.p_bar.update(n)

    def close_p_bar(self):
        self.p_bar.close()

    def _tqdm_dict(self, nth_process, _main_args_):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""
        model_func = list(_main_args_.search_config.keys())[
            nth_process % _main_args_.n_models
        ]
        return {
            "total": _main_args_.n_iter,
            "desc": "Thread " + str(nth_process) + " -> " + model_func.__name__,
            "position": nth_process,
            "leave": True,
        }


class ProgressBarLVL1(ProgressBarLVL0):
    def __init__(self):
        self.best_since_iter = 0

    def update_p_bar(self, n, _cand_):
        self.p_bar.update(n)
        self.p_bar.set_postfix(
            best_score=str(_cand_.score_best), best_since_iter=self.best_since_iter
        )
