# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import warnings


def try_ray_import():
    try:
        import ray

        if ray.is_initialized():
            rayInit = True
        else:
            rayInit = False
    except ImportError:
        warnings.warn("failed to import ray", ImportWarning)
        ray = None
        rayInit = False

    return ray, rayInit


class Distribution:
    def dist(self, optimizer_class, _main_args_, _opt_args_):
        ray, rayInit = try_ray_import()

        if rayInit:
            self.dist_ray(optimizer_class, _main_args_, _opt_args_, ray)
        else:
            self.dist_default(optimizer_class, _main_args_, _opt_args_)

    def dist_default(self, optimizer_class, _main_args_, _opt_args_):
        _optimizer_ = optimizer_class(_main_args_, _opt_args_)
        self.results, self.pos, self.scores, self.eval_times, self.iter_times, self.best_scores = (
            _optimizer_.search()
        )

    def dist_ray(self, optimizer_class, _main_args_, _opt_args_, ray):
        optimizer_class = ray.remote(optimizer_class)
        opts = [
            optimizer_class.remote(_main_args_, _opt_args_)
            for job in range(_main_args_.n_jobs)
        ]
        searches = [
            opt.search.remote(job, rayInit=True) for job, opt in enumerate(opts)
        ]
        self.results, self.pos, self.scores, self.eval_times, self.iter_times, self.best_scores = ray.get(
            searches
        )[
            0
        ]

        ray.shutdown()
