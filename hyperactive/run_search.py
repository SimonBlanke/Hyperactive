# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


from .distribution import (
    single_process,
    joblib_wrapper,
    multiprocessing_wrapper,
)


def run_search(process_info_dict):
    process_infos = list(process_info_dict.values())

    if len(process_infos) == 1:
        results_list = single_process(process_infos)
    else:
        results_list = multiprocessing_wrapper(process_infos)

    return results_list

