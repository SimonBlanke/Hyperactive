# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import collections
import numpy as np


def check_hyperactive_para(X, y, memory, random_state, verbosity):
    memory_list = ["short", "long", True, False]

    if not isinstance(X, np.ndarray):
        raise ValueError(r"Positional argument X must be of type numpy.ndarray")

    if not isinstance(y, np.ndarray):
        raise ValueError(r"Positional argument X must be of type numpy.ndarray")

    if memory not in memory_list:
        raise ValueError(
            r'Keyword argument memory must be "short", "long", True or False'
        )

    if not isinstance(random_state, int) and random_state is not False:
        raise ValueError(r"Keyword argument random_state must be of type int or False")

    if not isinstance(verbosity, int):
        raise ValueError(r"Keyword argument verbosity must be of type int")


def check_para(para, type):
    if not isinstance(para, type):
        raise ValueError(
            r"Keyword argument " + str(para) + "must be of type " + str(type)
        )


def check_search_para(
    search_config, max_time, n_iter, optimizer, n_jobs, scheduler, init_config
):
    scheduler_list = [None, "default", "smart"]

    if not isinstance(search_config, dict):
        raise ValueError(r"Positional argument search_config must be of type dict")
    elif isinstance(search_config, dict):
        _check_config(search_config)

    if not isinstance(max_time, (int, float)) and max_time is not None:
        raise ValueError(
            r"Keyword argument max_time must be of type int, float or None"
        )

    check_para(n_iter, int)
    check_para(optimizer, (dict, str))
    check_para(n_jobs, int)

    if scheduler not in scheduler_list:
        raise ValueError(
            r'Keyword argument scheduler must be None, "default" or "smart"'
        )

    if not isinstance(init_config, dict) and init_config is not None:
        raise ValueError(r"Keyword argument init_config must be of type dict or None")
    elif isinstance(init_config, dict):
        _check_config(init_config)


def _check_config(config):
    for key in config.keys():
        if not isinstance(key, collections.Callable):
            raise ValueError(r"Key in " + str(config) + " must be callable")
        if not isinstance(config[key], dict):
            raise ValueError(r"Value in " + str(config) + " must be of type dict")
