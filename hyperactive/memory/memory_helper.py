# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import sys
import glob
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


def reset_memory():
    if query_yes_no():
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
    model_hash = _get_model_hash(model)
    path = meta_path + str(model_hash)

    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(meta_path + str(model_hash))
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

    model1_hash = _get_model_hash(model1)
    model2_hash = _get_model_hash(model2)

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

    model1_hash = _get_model_hash(model1)
    model2_hash = _get_model_hash(model2)

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
    func_path_ = _get_model_hash(model) + "/"
    func_path = meta_path + func_path_

    feature_hash = _get_hash(X)
    label_hash = _get_hash(y)

    return func_path + (feature_hash + "_" + label_hash + "_.csv")


def _get_model_hash(model):
    return str(_get_hash(_get_func_str(model).encode("utf-8")))


def _get_func_str(func):
    return inspect.getsource(func)


def _get_hash(object):
    return hashlib.sha1(object).hexdigest()
