# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import inspect
import hashlib


def _get_func_str(func):
    return inspect.getsource(func)


def _get_hash(object):
    return hashlib.sha1(object).hexdigest()


def _get_model_hash(model):
    return str(_get_hash(_get_func_str(model).encode("utf-8")))


def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True
