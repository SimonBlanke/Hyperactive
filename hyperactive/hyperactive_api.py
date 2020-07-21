# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .general_optimizer import Optimizer


class Hyperactive:
    def __init__(
        self, X, y, random_state=None, verbosity=3, warnings=False, ext_warnings=False,
    ):
        self.X = X
        self.y = y
        self.opt = Optimizer(
            random_state=random_state,
            verbosity=verbosity,
            warnings=warnings,
            ext_warnings=ext_warnings,
        )

    def add_search(
        self,
        model,
        search_space,
        name=None,
        n_iter=10,
        function_parameter=None,
        optimizer="RandomSearch",
        n_jobs=1,
        init_para=[],
        memory="short",
    ):
        self.opt.add_search(
            objective_function=model,
            search_space=search_space,
            function_parameter={"features": self.X, "target": self.y},
            name=name,
            n_iter=n_iter,
            optimizer=optimizer,
            n_jobs=n_jobs,
            init_para=init_para,
            memory=memory,
        )

    def run(self, max_time=None):
        self.opt.run(max_time=max_time)

        # self.position_results = self.opt.position_results
        self.eval_times = self.opt.eval_times
        self.iter_times = self.opt.iter_times
        self.best_para = self.opt.best_para
        self.best_score = self.opt.best_score
