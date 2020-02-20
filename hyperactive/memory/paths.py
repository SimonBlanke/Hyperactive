# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import datetime


def get_meta_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/meta_data/"


def get_model_path(model_id):
    return model_id + "/"


def get_date_path(datetime):
    return "run_data/" + datetime + "/"


def get_datetime():
    return datetime.datetime.now().strftime("%d.%m.%Y - %H:%M:%S")
