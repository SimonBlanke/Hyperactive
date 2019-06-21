# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import tqdm
from .base import BaseOptimizer


class EvolutionStrategy_Optimizer(BaseOptimizer):
    def __init__(
        self,
        search_config,
        n_iter,
        metric="accuracy",
        memory=None,
        n_jobs=1,
        cv=5,
        verbosity=1,
        random_state=None,
        warm_start=False,
        generations=10,
        mutation_rate=0.5,
        crossover_rate=0.5,
    ):
        super().__init__(
            search_config,
            n_iter,
            metric,
            memory,
            n_jobs,
            cv,
            verbosity,
            random_state,
            warm_start,
        )

    def _search(self, n_process, X_train, y_train):
        hyperpara_indices = self._init_search(n_process, X_train, y_train)
        self._set_random_seed(n_process)
        self.n_steps = self._set_n_steps(n_process)

        hyperpara_dict = self.search_space_inst.pos_dict2values_dict(hyperpara_indices)

        score, train_time, sklearn_model = self.model.train_model(
            hyperpara_dict, X_train, y_train
        )

        self.best_score = score
        self.best_hyperpara_dict = hyperpara_dict
        self.best_model = sklearn_model

        for i in tqdm.tqdm(
            range(self.n_steps),
            desc=str(self.model_str),
            position=n_process,
            leave=False,
        ):
            pass
