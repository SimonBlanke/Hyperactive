# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm.auto import tqdm


class ProgressBarLVL0:
    def __init__(self):
        pass

    def init_p_bar(self, nth_process, n_iter, obj_func):
        pass

    def update_p_bar(self, n, score_best):
        pass

    def close_p_bar(self):
        pass

    def _tqdm_dict(self, nth_process, n_iter, obj_func):
        pass


class ProgressBarLVL1:
    def __init__(self):
        self.best_since_iter = 0

    def init_p_bar(self, nth_process, n_iter, obj_func):
        self._tqdm = tqdm(**self._tqdm_dict(nth_process, n_iter, obj_func))

    def update_p_bar(self, n, score_best):
        self._tqdm.update(n)
        self._tqdm.set_postfix(
            best_score=str(score_best), best_since_iter=self.best_since_iter
        )

    def close_p_bar(self):
        self._tqdm.close()

    def _tqdm_dict(self, nth_process, n_iter, obj_func):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""
        return {
            "total": n_iter,
            "desc": "Process " + str(nth_process) + " -> " + obj_func.__name__,
            "position": nth_process,
            "leave": True,
        }

