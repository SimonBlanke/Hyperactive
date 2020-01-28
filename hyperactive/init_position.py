# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .util import sort_for_best
import numpy as np


class InitSearchPosition:
    def __init__(self, space, model, _main_args_):
        self._space_ = space
        self._model_ = model
        self.init_config = _main_args_.init_config
        self.X = _main_args_.X
        self.y = _main_args_.y

    def _warm_start(self):
        pos = []

        for hyperpara_name in self._space_.search_space.keys():
            if hyperpara_name not in list(self._space_.init_para.keys()):
                search_position = self._space_.get_random_pos_scalar(hyperpara_name)

            else:
                search_position = self._space_.search_space[hyperpara_name].index(
                    self._space_.init_para[hyperpara_name]
                )
            pos.append(search_position)

        return np.array(pos)

    def _set_start_pos(self, _info_):
        if self._space_.init_type == "warm_start":
            _info_.warm_start()
            pos = self._warm_start()
        elif self._space_.init_type == "scatter_init":
            _info_.scatter_start()
            pos = self._scatter_init()
        else:
            _info_.random_start()
            pos = self._space_.get_random_pos()

        return pos

    def _scatter_init(self):
        pos_list = []
        for _ in range(self._space_.init_para["scatter_init"]):
            pos = self._space_.get_random_pos()
            pos_list.append(pos)

        pos_best_list, score_best_list = self._scatter_train(pos_list)
        pos_best_sorted, _ = sort_for_best(pos_best_list, score_best_list)

        return pos_best_sorted[0]

    def _scatter_train(self, pos_list):
        pos_best_list = []
        score_best_list = []

        X, y = self._get_random_sample(self.X, self.y)

        for pos in pos_list:
            para = self._space_.pos2para(pos)
            score, eval_time = self._model_.train_model(para)

            pos_best_list.append(pos)
            score_best_list.append(score)

        return pos_best_list, score_best_list

    def _get_random_sample(self, X, y):
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            n_samples = int(X.shape[0] / self._space_.init_para["scatter_init"])

            idx = np.random.choice(np.arange(len(X)), n_samples, replace=False)

            X_sample = X[idx]
            y_sample = y[idx]

            return X_sample, y_sample
