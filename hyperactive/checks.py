# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import collections
import numpy as np


def _check_objective_function(value):
    if not isinstance(value, collections.Callable):
        raise ValueError(r"objective_function must be callable")


def _check_function_parameter(value):
    if not isinstance(value, dict):
        raise ValueError(r"function_parameter must be of type dict")


def _check_search_space(value):
    if not isinstance(value, dict):
        raise ValueError(r"search_space must be of type dict")


def _check_memory(value):
    memory_list = ["short", "long", False]
    if value not in memory_list:
        raise ValueError(r"memory must be 'short', 'long' or False")


def _check_optimizer(value):
    if not isinstance(value, (dict, str)):
        raise ValueError(r"optimizer must be of type dict or str")


def _check_n_iter(value):
    if not isinstance(value, int):
        raise ValueError(r"n_iter must be of type int")


def _check_n_jobs(value):
    if not isinstance(value, int):
        raise ValueError(r"n_jobs must be of type int")


def _check_init_para(value):
    if not isinstance(value, list) and value is not None:
        raise ValueError(r"init_para must be of type list or None")


def _check_distribution(value):
    if not isinstance(value, dict) and value is not None:
        raise ValueError(r"distribution must be of type dict or None")


arguments = {
    "objective_function": _check_objective_function,
    "function_parameter": _check_function_parameter,
    "search_space": _check_search_space,
    "memory": _check_memory,
    "optimizer": _check_optimizer,
    "n_iter": _check_n_iter,
    "n_jobs": _check_n_jobs,
    "init_para": _check_init_para,
    "distribution": _check_distribution,
}


def check_kwargs(kwargs):
    for keyword in kwargs:
        if keyword not in arguments:
            raise TypeError(
                "add_search got an unexpected keyword argument " + str(keyword)
            )

        value = kwargs[keyword]
        check_function = arguments[keyword]
        check_function(value)

