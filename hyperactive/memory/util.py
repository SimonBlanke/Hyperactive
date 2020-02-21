# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import glob
import dill
import inspect
import hashlib


def get_func_str(func):
    return inspect.getsource(func)


def get_hash(object):
    return hashlib.sha1(object).hexdigest()


def get_model_id(model):
    return str(get_hash(get_func_str(model).encode("utf-8")))


def _get_pkl_hash(hash, model_path):
    paths = glob.glob(model_path + hash + "*.pkl")

    return paths


def _hash2obj(search_space, model_path):
    hash2obj_dict = {}
    para_hash_list = _get_para_hash_list(search_space)

    for para_hash in para_hash_list:
        obj = _read_dill(para_hash, model_path)
        hash2obj_dict[para_hash] = obj

    return hash2obj_dict


def _read_dill(value, model_path):
    paths = _get_pkl_hash(value, model_path)
    for path in paths:
        with open(path, "rb") as fp:
            value = dill.load(fp)
            value = dill.loads(value)
            break

    return value


def _get_para_hash_list(search_space):
    para_hash_list = []
    for key in search_space.keys():
        values = search_space[key]

        for value in values:
            if (
                not isinstance(value, int)
                and not isinstance(value, float)
                and not isinstance(value, str)
            ):

                para_dill = dill.dumps(value)
                para_hash = get_hash(para_dill)
                para_hash_list.append(para_hash)

    return para_hash_list


"""
def is_sha1(maybe_sha):
    if len(maybe_sha) != 40:
        return False
    try:
        sha_int = int(maybe_sha, 16)
    except ValueError:
        return False
    return True
"""
