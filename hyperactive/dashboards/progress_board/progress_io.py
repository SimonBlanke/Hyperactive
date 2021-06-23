# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import pandas as pd


class ProgressIO:
    def __init__(self, path, hide_progress_data=True, verbosity=True, warnings=True):
        self.path = path
        self.verbosity = verbosity
        self.warnings = warnings

        if hide_progress_data:
            self.csv = ".csv~"
        else:
            self.csv = ".csv"

    def get_filter_file_path(self, search_id):
        return self.path + "/filter_" + search_id + ".csv"

    def get_progress_data_path(self, search_id):
        return self.path + "/progress_data_" + search_id + self.csv

    def load_filter(self, search_id):
        path = self.get_filter_file_path(search_id)
        if os.path.isfile(path):
            if self.verbosity:
                print("Load filter file from path:", path)
            return pd.read_csv(path)
        else:
            if self.warnings:
                print("Warning: Filter file not found in:", path)
            return None

    def load_progress(self, search_id):
        path = self.get_progress_data_path(search_id)
        if os.path.isfile(path):
            if self.verbosity:
                print("Load progress data file from path:", path)
            return pd.read_csv(path)
        else:
            if self.warnings:
                print("Warning: Progress data not found in:", path)
            return None

    def remove_filter(self, search_id):
        path = self.get_filter_file_path(search_id)
        if os.path.isfile(path):
            os.remove(path)
            if self.verbosity:
                print("Remove filter file from path:", path)

    def remove_progress(self, search_id):
        path = self.get_progress_data_path(search_id)
        if os.path.isfile(path):
            os.remove(path)
            if self.verbosity:
                print("Remove progress data file from path:", path)

    def create_filter(self, search_id, search_space):
        path = self.get_filter_file_path(search_id)
        self.remove_filter(search_id)

        indices = list(search_space.keys()) + ["score"]
        filter_dict = {
            "parameter": indices,
            "lower bound": "---",
            "upper bound": "---",
        }

        df = pd.DataFrame(filter_dict)
        df.to_csv(path, index=None)