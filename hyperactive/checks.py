# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import collections
import numpy as np


def check_hyperactive_para(X, y, memory, random_state, verbosity):
    memory_list = ["short", "long", True, False]

    if not isinstance(X, np.ndarray):
        raise ValueError(r'Positional argument X must be of type numpy.ndarray')

    if not isinstance(y, np.ndarray):
        raise ValueError(r'Positional argument X must be of type numpy.ndarray')

    if memory not in memory_list:
        raise ValueError(r'Keyword argument memory must be "short", "long", True or False')

    if not isinstance(random_state, int) and not random_state == False:
        raise ValueError(r'Keyword argument random_state must be of type int or False')

    if not isinstance(verbosity, int):
        raise ValueError(r'Keyword argument verbosity must be of type int')


def check_search_para(search_config, max_time, n_iter, optimizer, n_jobs, init_config):

    if not isinstance(search_config, dict):
        raise ValueError(r'Positional argument search_config must be of type dict')
    elif isinstance(search_config, dict):
        _check_search_config(search_config)

    if not isinstance(max_time, (int, float)) and not max_time == None:
        raise ValueError(r'Keyword argument max_time must be of type int, float or None')

    if not isinstance(n_iter, int):
        raise ValueError(r'Keyword argument n_iter must be of type int')

    if not isinstance(optimizer, dict) and not isinstance(optimizer, str):
        raise ValueError(r'Keyword argument optimizer must be of type str or dict')

    if not isinstance(n_jobs, int):
        raise ValueError(r'Keyword argument n_jobs must be of type int')

    if not isinstance(init_config, dict) and not init_config == None:
        raise ValueError(r'Keyword argument init_config must be of type dict or None')
    elif isinstance(init_config, dict):
        _check_init_config(init_config)

    
def _check_search_config(search_config):
    for key in search_config.keys():
        if not isinstance(key, collections.Callable):
            raise ValueError(r'Key in search_config must be callable')
        if not isinstance(search_config[key], dict):
            raise ValueError(r'Value in search_config must be of type dict')

def _check_init_config(init_config):
    for key in init_config.keys():
        if not isinstance(key, collections.Callable):
            raise ValueError(r'Key in init_config must be callable')
        if not isinstance(init_config[key], dict):
            raise ValueError(r'Value in init_config must be of type dict')
        
