# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm.auto import tqdm


class ProgressBar:
    def init_p_bar(self, nth_process, n_iter, obj_func):
        pass

    def update_p_bar(self, n, _cand_):
        pass

    def close_p_bar(self):
        pass


class ProgressBarLVL0(ProgressBar):
    def __init__(self):
        self.best_since_iter = 0

    def init_p_bar(self, nth_process, n_iter, obj_func):
        self.p_bar = tqdm(**self._tqdm_dict(nth_process, n_iter, obj_func))

    def update_p_bar(self, n, _cand_):
        self.p_bar.update(n)

    def close_p_bar(self):
        self.p_bar.close()

    def _tqdm_dict(self, nth_process, n_iter, obj_func):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""
        return {
            "total": n_iter,
            "desc": "Thread " + str(nth_process) + " -> " + obj_func.__name__,
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
