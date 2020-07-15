# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os


def meta_data_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/"
