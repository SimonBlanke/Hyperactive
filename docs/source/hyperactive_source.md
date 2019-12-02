## Hyperactive

```python
class Hyperactive:
    def __init__(self, X, y, memory=True, random_state=1, verbosity=3, warnings=False):
        self.X = X
        self._main_args_ = MainArgs(X, y, memory, random_state, verbosity)

        if not warnings:
            stop_warnings()

        self.optimizer_dict = {
            "HillClimbing": HillClimbingOptimizer,
            "StochasticHillClimbing": StochasticHillClimbingOptimizer,
            "TabuSearch": TabuOptimizer,
            "RandomSearch": RandomSearchOptimizer,
            "RandomRestartHillClimbing": RandomRestartHillClimbingOptimizer,
            "RandomAnnealing": RandomAnnealingOptimizer,
            "SimulatedAnnealing": SimulatedAnnealingOptimizer,
            "StochasticTunneling": StochasticTunnelingOptimizer,
            "ParallelTempering": ParallelTemperingOptimizer,
            "ParticleSwarm": ParticleSwarmOptimizer,
            "EvolutionStrategy": EvolutionStrategyOptimizer,
            "Bayesian": BayesianOptimizer,
        }

    def search(
        self,
        search_config,
        n_iter=10,
        max_time=None,
        optimizer="RandomSearch",
        n_jobs=1,
        warm_start=False,
        scatter_init=False,
    ):

        start_time = time.time()

        self._main_args_.search_args(
            search_config, max_time, n_iter, optimizer, n_jobs, warm_start, scatter_init
        )
        self._opt_args_ = Arguments(self._main_args_.opt_para)
        optimizer_class = self.optimizer_dict[self._main_args_.optimizer]

        try:
            import ray

            if ray.is_initialized():
                ray_ = True
            else:
                ray_ = False
        except ImportError:
            warnings.warn("failed to import ray", ImportWarning)
            ray_ = False

        if ray_:
            optimizer_class = ray.remote(optimizer_class)
            opts = [
                optimizer_class.remote(self._main_args_, self._opt_args_)
                for job in range(self._main_args_.n_jobs)
            ]
            searches = [
                opt.search.remote(job, ray_=ray_) for job, opt in enumerate(opts)
            ]
            ray.get(searches)
        else:
            self._optimizer_ = optimizer_class(self._main_args_, self._opt_args_)
            self._optimizer_.search()

        self.results_params = self._optimizer_.results_params
        self.results_models = self._optimizer_.results_models

        self.pos_list = self._optimizer_.pos_list
        self.score_list = self._optimizer_.score_list

        self.total_time = time.time() - start_time

    def get_total_time(self):
        return self.total_time

    def get_eval_time(self):
        return self._optimizer_.eval_time

    def save_report(self):
        pass
```
