# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import sys
import json
import dill
import shutil
import pathlib
from fnmatch import fnmatch

import numpy as np
import pandas as pd

from .util import get_hash, get_model_id, get_func_str
from .paths import get_meta_path, get_meta_data_name


meta_path = get_meta_path()

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


def get_best_model(X, y):
    meta_data_paths = []
    pattern = get_meta_data_name(X, y)

    for path, subdirs, files in os.walk(meta_path):
        for name in files:
            if fnmatch(name, pattern):
                meta_data_paths.append(pathlib.PurePath(path, name))

    score_best = -np.inf

    for path in meta_data_paths:
        path = str(path)
        meta_data = pd.read_csv(path)
        scores = meta_data["_score_"].values

        # score_mean = scores.mean()
        # score_std = scores.std()
        score_max = scores.max()
        # score_min = scores.min()

        if score_max > score_best:
            score_best = score_max

            model_path = path.rsplit("dataset_id:", 1)[0]

            obj_func_path = model_path + "objective_function.pkl"
            search_space_path = model_path + "search_space.pkl"

            with open(obj_func_path, "rb") as fp:
                obj_func = dill.load(fp)

            with open(search_space_path, "rb") as fp:
                search_space = dill.load(fp)

            para_names = list(search_space.keys())

            best_para = meta_data[meta_data["_score_"] == score_max]
            best_para = best_para[para_names].iloc[0]

            best_para = best_para.to_dict()

        return (score_best, {obj_func: search_space}, {obj_func: best_para})


def reset_memory(force_true=False):
    if force_true:
        _reset_memory()
    elif query_yes_no():
        _reset_memory()


def _reset_memory():
    dirs = next(os.walk(meta_path))[1]
    for dir in dirs:
        shutil.rmtree(meta_path + dir)

    with open(meta_path + "model_connections.json", "w") as f:
        json.dump({}, f, indent=4)

    print("Memory reset successful")


def query_yes_no():
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    question = "Delete the entire long term memory?"

    while True:
        sys.stdout.write(question + " [y/n] ")
        choice = input().lower()
        if choice == "":
            return False
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def delete_model(model):
    model_hash = get_model_id(model)
    path = meta_path + "model_id:" + str(model_hash)

    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)
        print("Model data successfully removed")
    else:
        print("Model data not found in memory")


def delete_model_dataset(model, X, y):
    csv_file = _get_file_path(model, X, y)

    if os.path.exists(csv_file):
        os.remove(csv_file)
        print("Model data successfully removed")
    else:
        print("Model data not found in memory")


def connect_model_IDs(model1, model2):
    # do checks if search space has same dim

    with open(meta_path + "model_connections.json") as f:
        data = json.load(f)

    model1_hash = get_model_id(model1)
    model2_hash = get_model_id(model2)

    if model1_hash in data:
        key_model = model1_hash
        value_model = model2_hash
        data = _connect_key2value(data, key_model, value_model)
    else:
        data[model1_hash] = [model2_hash]
        print("IDs successfully connected")

    if model2_hash in data:
        key_model = model2_hash
        value_model = model1_hash
        data = _connect_key2value(data, key_model, value_model)
    else:
        data[model2_hash] = [model1_hash]
        print("IDs successfully connected")

    with open(meta_path + "model_connections.json", "w") as f:
        json.dump(data, f, indent=4)


def _connect_key2value(data, key_model, value_model):
    if value_model in data[key_model]:
        print("IDs of models are already connected")
    else:
        data[key_model].append(value_model)
        print("IDs successfully connected")

    return data


def _split_key_value(data, key_model, value_model):
    if value_model in data[key_model]:
        data[key_model].remove(value_model)

        if len(data[key_model]) == 0:
            del data[key_model]
        print("ID connection successfully deleted")
    else:
        print("IDs of models are not connected")

    return data


def split_model_IDs(model1, model2):
    # TODO: do checks if search space has same dim

    with open(meta_path + "model_connections.json") as f:
        data = json.load(f)

    model1_hash = get_model_id(model1)
    model2_hash = get_model_id(model2)

    if model1_hash in data:
        key_model = model1_hash
        value_model = model2_hash
        data = _split_key_value(data, key_model, value_model)
    else:
        print("IDs of models are not connected")

    if model2_hash in data:
        key_model = model2_hash
        value_model = model1_hash
        data = _split_key_value(data, key_model, value_model)
    else:
        print("IDs of models are not connected")

    with open(meta_path + "model_connections.json", "w") as f:
        json.dump(data, f, indent=4)


def _get_file_path(model, X, y):
    func_path_ = "model_id:" + get_model_id(model) + "/"
    func_path = meta_path + func_path_

    feature_hash = get_hash(X)
    label_hash = get_hash(y)

    return func_path + (feature_hash + "_" + label_hash + "_.csv")
