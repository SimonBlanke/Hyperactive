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

    def init_p_bar(self, _cand_, _core_):
        self.p_bar = tqdm(**self._tqdm_dict(_cand_, _core_))

    def update_p_bar(self, n, _cand_):
        self.p_bar.update(n)

    def close_p_bar(self):
        self.p_bar.close()

    def _tqdm_dict(self, _cand_, _core_):
        """Generates the parameter dict for tqdm in the iteration-loop of each optimizer"""
        return {
            "total": _core_.n_iter,
            "desc": "Thread "
            + str(_cand_.nth_process)
            + " -> "
            + _cand_._model_.func_.__name__,
            "position": _cand_.nth_process,
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
