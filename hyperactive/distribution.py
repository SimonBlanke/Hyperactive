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


def dist(optimizer_class, _main_args_, _opt_args_):
    ray, rayInit = try_ray_import()

    if rayInit:
        dist_ray(optimizer_class, _main_args_, _opt_args_)
    else:
        dist_default(optimizer_class, _main_args_, _opt_args_)


def dist_default(optimizer_class, _main_args_, _opt_args_):
    _optimizer_ = optimizer_class(_main_args_, _opt_args_)
    params_results, pos_list, score_list = (
        _optimizer_.search()
    )

    # print("params_results", params_results)

def dist_ray(optimizer_class, _main_args_, _opt_args_):
    optimizer_class = ray.remote(optimizer_class)
    opts = [
        optimizer_class.remote(_main_args_, _opt_args_)
        for job in range(_main_args_.n_jobs)
    ]
    searches = [
        opt.search.remote(job, rayInit=rayInit) for job, opt in enumerate(opts)
    ]
    params_results, pos_list, score_list = ray.get(searches)[0]

    # print("params_results", params_results)

    ray.shutdown()