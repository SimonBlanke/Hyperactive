# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import os
import contextlib
import pandas as pd
from filelock import FileLock


@contextlib.contextmanager
def atomic_overwrite(filename):
    # from: https://stackoverflow.com/questions/42409707/pandas-to-csv-overwriting-prevent-data-loss
    temp = filename + "~"
    with open(temp, "w") as f:
        yield f
    os.rename(temp, filename)  # this will only happen if no exception was raised


class DataIO:
    def __init__(self, path, drop_duplicates):
        self.path = path
        self.replace_existing = False
        self.drop_duplicates = drop_duplicates

        if self.replace_existing:
            self.mode = "w"
        else:
            self.mode = "a"

    def _save_dataframe(self, dataframe, io_wrap):
        if self.drop_duplicates:
            dataframe.drop_duplicates(subset=self.drop_duplicates, inplace=True)

        dataframe.to_csv(io_wrap, index=False, header=not io_wrap.tell())

    def atomic_write(self, dataframe, path, replace_existing):
        self.replace_existing = replace_existing

        with atomic_overwrite(path) as io_wrap:
            self._save_dataframe(dataframe, io_wrap)

    def locked_write(self, dataframe, path):

        lock = FileLock(path + ".lock~")
        with lock:
            with open(path, self.mode) as io_wrap:
                self._save_dataframe(dataframe, io_wrap)

        """
        import fcntl

        with open(path, self.mode) as io_wrap:
            fcntl.flock(io_wrap, fcntl.LOCK_EX)
            self._save_dataframe(dataframe, io_wrap)
            fcntl.flock(io_wrap, fcntl.LOCK_UN)
        """

    def load(self, path):
        if os.path.isfile(self.path) and os.path.getsize(self.path) > 0:
            return pd.read_csv(self.path)


class DataCollector:
    def __init__(self, path, drop_duplicates=False):
        self.path = path
        self.drop_duplicates = drop_duplicates

        self.path2file = path.rsplit("/", 1)[0] + "/"
        self.file_name = path.rsplit("/", 1)[1]

        self.io = DataIO(path, drop_duplicates)

    def load(self):
        return self.io.load(self.path)

    def append(self, dictionary):
        dataframe = pd.DataFrame(dictionary, index=[0])
        self.io.locked_write(dataframe, self.path)

    def save(self, dataframe, replace_existing=False):
        self.io.atomic_write(dataframe, self.path, replace_existing)
