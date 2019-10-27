# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import abc
from tqdm.auto import tqdm

from .util import sort_for_best


class Verbosity(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def print_start_point(self, _cand_):
        pass

    @abc.abstractmethod
    def print_start_points(self, _cand_):
        pass

    def init_p_bar(self, _cand_, _core_):
        pass

    def update_p_bar(self, n, _cand_):
        pass

    def close_p_bar(self):
        pass

    def _tqdm_dict(self, _cand_):
        pass


class VerbosityLVL0(Verbosity):
    def __init__(self):
        pass

    def print_start_point(self, _cand_):
        return _cand_._get_warm_start()

    def print_start_points(self, _cand_list, _core_):
        start_point_list = []
        score_best_list = []
        model_best_list = []
        results = {}

        for _cand_ in _cand_list:
            model_best = _cand_.model_best
            score_best = _cand_.score_best
            start_point = _cand_._get_warm_start()

            results[score_best] = start_point

            start_point_list.append(start_point)
            score_best_list.append(score_best)
            model_best_list.append(model_best)

        start_point_sorted, score_best_sorted = sort_for_best(
            start_point_list, score_best_list
        )

        model_best_sorted, score_best_sorted = sort_for_best(
            model_best_list, score_best_list
        )

        return score_best_sorted, model_best_sorted, results


class VerbosityLVL1(VerbosityLVL0):
    def __init__(self):
        pass

    def print_start_point(self, _cand_):
        start_point = _cand_._get_warm_start()
        print("\nbest para =", start_point)
        print("score     =", _cand_.score_best)

        return start_point

    def print_start_points(self, _cand_list, _core_):
        start_point_list = []
        score_best_list = []
        model_best_list = []
        results = {}

        for _cand_ in _cand_list:
            model_best = _cand_.model_best
            score_best = _cand_.score_best
            start_point = _cand_._get_warm_start()

            results[score_best] = start_point

            start_point_list.append(start_point)
            score_best_list.append(score_best)
            model_best_list.append(model_best)

        start_point_sorted, score_best_sorted = sort_for_best(
            start_point_list, score_best_list
        )

        model_best_sorted, score_best_sorted = sort_for_best(
            model_best_list, score_best_list
        )

        for i in range(int(_core_.n_jobs / 2)):
            print("\n")
        print("\nList of start points (best first):\n")
        for start_point, score_best in zip(start_point_sorted, score_best_sorted):
            print("best para =", start_point)
            print("score     =", score_best, "\n")

        return score_best_sorted, model_best_sorted, results


class VerbosityLVL2(VerbosityLVL1):
    def __init__(self):
        pass

    def init_p_bar(self, _cand_, _core_):
        self.p_bar = tqdm(**self._tqdm_dict(_cand_, _core_))

    def update_p_bar(self, n, _cand_):
        self.p_bar.update(n)
        self.p_bar.set_postfix(best_score=str(_cand_.score_best))

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


class VerbosityLVL10(VerbosityLVL0):
    def __init__(self):
        pass

    def start_search(self):
        print("")

    def get_search_path(self):
        pass
