# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import datetime

from .util import get_hash


def get_meta_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/meta_data/"


def get_model_path(model_id):
    return "model_id:" + model_id + "/"


def get_date_path(datetime):
    return "run_data/" + datetime + "/"


def get_datetime():
    return datetime.datetime.now().strftime("%d.%m.%Y - %H:%M:%S:%f")


def get_meta_data_name(X, y):
    feature_hash = get_hash(X)
    label_hash = get_hash(y)

    return "dataset_id:" + feature_hash + "_" + label_hash + "_.csv"
