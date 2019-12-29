# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from tqdm.auto import tqdm


class Verbosity:
    def init_p_bar(self, _cand_, _core_):
        pass

    def update_p_bar(self, n, _cand_):
        pass

    def close_p_bar(self):
        pass

    def warm_start(self):
        pass

    def scatter_start(self):
        pass

    def random_start(self):
        pass

    def load_meta_data(self):
        pass

    def no_meta_data(self, model_func):
        pass

    def load_samples(self, para):
        pass


class VerbosityLVL0(Verbosity):
    def __init__(self):
        pass

    def print_start_point(self, _cand_):
        return _cand_._get_warm_start()


class VerbosityLVL1(VerbosityLVL0):
    def __init__(self):
        pass

    def print_start_point(self, _cand_):
        start_point = _cand_._get_warm_start()
        print("best para =", start_point)
        print("score     =", _cand_.score_best, "\n")

        return start_point

    def warm_start(self):
        print("Set warm start")

    def scatter_start(self):
        print("Set scatter init")

    def random_start(self):
        print("Set random start position")

    def load_meta_data(self):
        print("Loading meta data successful", end="\r")

    def no_meta_data(self, model_func):
        print("No meta data found for", model_func.__name__, "function")

    def load_samples(self, para):
        print("Loading meta data successful:", len(para), "samples found")


class VerbosityLVL2(VerbosityLVL1):
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


class VerbosityLVL3(VerbosityLVL2):
    def __init__(self):
        self.best_since_iter = 0

    def update_p_bar(self, n, _cand_):
        self.p_bar.update(n)
        self.p_bar.set_postfix(
            best_score=str(_cand_.score_best), best_since_iter=self.best_since_iter
        )
