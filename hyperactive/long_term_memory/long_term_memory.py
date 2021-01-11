# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import dill
import pandas as pd


def meta_data_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/data/"


class LongTermMemory:
    def __init__(self, model_name, path=None, verbosity=None):
        self.model_name = model_name

        if path is None:
            self.model_path = meta_data_path() + str(self.model_name) + ".pkl"
        else:
            self.model_path = path + str(self.model_name) + ".pkl"

        self.n_old_samples = 0
        self.n_new_samples = 0

    def _pkl_valid(self):
        return (
            os.path.isfile(self.model_path)
            and os.path.getsize(self.model_path) > 0
        )

    def load(self):
        if self._pkl_valid():
            print("Reading in long term memory ...", end="\r")

            with open(self.model_path, "rb") as handle:
                self.results_old = dill.load(handle)

                self.n_old_samples = len(self.results_old)

            print(
                "Reading long term memory was successful:",
                self.n_old_samples,
                "samples found",
            )

            return self.results_old

    def save(self, dataframe):
        if self._pkl_valid():
            with open(self.model_path, "rb") as handle:
                self.results_old = dill.load(handle)

            self.n_old_samples = len(self.results_old)

            dataframe = (
                pd.concat([self.results_old, dataframe])
                .drop_duplicates(keep="last")
                .reset_index(drop=True)
            )
        self.n_new_samples = len(dataframe)

        print("Saving long term memory ...", end="\r")

        with open(self.model_path, "wb") as handle:
            dill.dump(dataframe, handle, protocol=dill.HIGHEST_PROTOCOL)

        print(
            "Saving long term memory was successful:",
            self.n_new_samples - self.n_old_samples,
            "new samples found",
        )
