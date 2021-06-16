# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import os
import uuid
import numpy as np
import pandas as pd

from ..data_tools import DataCollector


class ProgressBoard:
    def __init__(self, filter_file=None):
        self.filter_file = filter_file

        self.uuid = uuid.uuid4().hex
        self.paths_list = []

    def _create_filter_file(self, search_id, search_space):
        filter_path = "./filter_" + search_id + ".csv"
        if os.path.isfile(filter_path):
            os.remove(filter_path)

        indices = list(search_space.keys()) + ["score"]
        filter_dict = {
            "parameter": indices,
            "lower bound": "lower",
            "upper bound": "upper",
        }

        df = pd.DataFrame(filter_dict)
        df.to_csv(filter_path, index=None)

    def init_paths(self, search_id, search_space):
        progress_data_path = "./progress_data_" + search_id + ".csv~"

        if os.path.isfile(progress_data_path):
            os.remove(progress_data_path)

        data_c = DataCollector(progress_data_path)

        self.paths_list.append(search_id)
        if self.filter_file:
            self._create_filter_file(search_id, search_space)

        return data_c

    def open_dashboard(self):
        abspath = os.path.abspath(__file__)
        dir_ = os.path.dirname(abspath)

        paths = " ".join(self.paths_list)
        open_streamlit = "streamlit run " + dir_ + "/run_streamlit.py " + paths

        # from: https://stackoverflow.com/questions/7574841/open-a-terminal-from-python
        os.system("gnome-terminal -e 'bash -c \" " + open_streamlit + " \"'")
