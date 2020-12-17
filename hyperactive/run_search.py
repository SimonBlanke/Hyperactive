# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .distribution import (
    single_process,
    joblib_wrapper,
    multiprocessing_wrapper,
)

dist_dict = {
    "joblib": joblib_wrapper,
    "multiprocessing": multiprocessing_wrapper,
}


def _get_distribution(distribution):
    if hasattr(distribution, "__call__"):
        return distribution
    elif isinstance(distribution, dict):
        dist_key = list(distribution.keys())[0]
        dist_paras = list(distribution.values())[0]

        return dist_dict[dist_key](**dist_paras)
    elif isinstance(distribution, str):
        return dist_dict[distribution]


def run_search(search_processes_infos, distribution):
    process_infos = list(search_processes_infos.values())

    if len(process_infos) == 1:
        results_list = single_process(process_infos)
    else:

        distribution = _get_distribution(distribution)
        results_list = distribution(process_infos)

    return results_list

