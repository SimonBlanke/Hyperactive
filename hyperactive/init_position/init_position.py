# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np


class InitSearchPosition:
    def __init__(self, space, model, warm_start, scatter_init):
        self._space_ = space
        self._model_ = model
        self.warm_start = warm_start
        self.scatter_init = scatter_init

        if self.warm_start:
            self.n_warm_start_keys = len(list(self.warm_start.keys()))
        else:
            self.n_warm_start_keys = 0

    def _set_start_pos(self, nth_process, X, y):
        if self.warm_start and self.scatter_init:
            if self.n_warm_start_keys > nth_process:
                pos = self._create_warm_start(nth_process)
            else:
                pos = self._scatter_init(nth_process, X, y)

        elif self.warm_start:
            if self.n_warm_start_keys > nth_process:
                pos = self._create_warm_start(nth_process)
            else:
                pos = self._space_.get_random_pos()

        elif self.scatter_init:
            pos = self._scatter_init(nth_process, X, y)

        else:
            pos = self._space_.get_random_pos()

        return pos

    def _scatter_init(self, nth_process, X, y):
        pos_list = []
        for _ in range(self.scatter_init):
            pos = self._space_.get_random_pos()
            pos_list.append(pos)

        pos_best_list, score_best_list = self._scatter_train(X, y, pos_list)

        pos_best_sorted, _ = self._sort_for_best(pos_best_list, score_best_list)

        nth_best_pos = nth_process - self.n_warm_start_keys

        return pos_best_sorted[nth_best_pos]

    def _sort_for_best(self, sort, sort_by):
        sort = np.array(sort)
        sort_by = np.array(sort_by)

        index_best = list(sort_by.argsort()[::-1])

        sort_sorted = sort[index_best]
        sort_by_sorted = sort_by[index_best]

        return sort_sorted, sort_by_sorted

    def _scatter_train(self, X, y, pos_list):
        pos_best_list = []
        score_best_list = []

        X, y = self._get_random_sample(X, y)

        for pos in pos_list:
            para = self._space_.pos2para(pos)
            score, _ = self._model_.train_model(para, X, y)

            pos_best_list.append(pos)
            score_best_list.append(score)

        return pos_best_list, score_best_list

    def _get_random_sample(self, X, y):
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            n_samples = int(X.shape[0] / self.scatter_init)

            idx = np.random.choice(np.arange(len(X)), n_samples, replace=False)

            X_sample = X[idx]
            y_sample = y[idx]

            return X_sample, y_sample
