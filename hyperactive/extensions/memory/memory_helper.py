# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import json
import shutil
import hashlib
import inspect


current_path = os.path.realpath(__file__)
meta_learn_path, _ = current_path.rsplit("/", 1)
meta_path = meta_learn_path + "/meta_data/"

"""
def get_best_models(X, y):
    # TODO: model_dict   key:model   value:score

    return model_dict


def get_model_search_config(model):
    # TODO
    return search_config


def get_model_init_config(model):
    # TODO
    return init_config
"""


def delete_model(model):
    model_hash = _get_model_hash(model)
    path = meta_path + str(model_hash)

    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(meta_path + str(model_hash))
        print("Model data successfully removed")
    else:
        print("Model data not found in memory")


def delete_model_dataset(model, X, y):
    csv_file = _get_file_path(model, X, y)
    print("csv_file", csv_file)

    if os.path.exists(csv_file):
        os.remove(csv_file)
        print("Model data successfully removed")
    else:
        print("Model data not found in memory")


def merge_model_hashes(model1, model2):
    # do checks if search space has same dim

    with open(meta_path + "model_connections.json") as f:
        data = json.load(f)

    model1_hash = _get_model_hash(model1)
    model2_hash = _get_model_hash(model2)

    models_dict = {str(model1_hash): str(model2_hash)}
    data.update(models_dict)

    with open(meta_path + "model_connections.json", "w") as f:
        json.dump(data, f)


def split_model_hashes(model1, model2):
    # TODO: do checks if search space has same dim

    with open(meta_path + "model_connections.json") as f:
        data = json.load(f)

    model1_hash = _get_model_hash(model1)
    model2_hash = _get_model_hash(model2)

    if model1_hash in data.keys():
        del data[model1_hash]
    if model2_hash in data.keys():
        del data[model2_hash]

    with open(meta_path + "model_connections.json", "w") as f:
        json.dump(data, f)


def _get_file_path(model, X, y):
    func_path_ = _get_model_hash(model) + "/"
    func_path = meta_path + func_path_

    feature_hash = _get_hash(X)
    label_hash = _get_hash(y)

    return func_path + (feature_hash + "_" + label_hash + "_.csv")


def _get_model_hash(model):
    return _get_hash(_get_func_str(model).encode("utf-8"))


def _get_func_str(func):
    return inspect.getsource(func)


def _get_hash(object):
    return hashlib.sha1(object).hexdigest()
