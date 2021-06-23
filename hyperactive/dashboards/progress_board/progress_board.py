# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import os
import uuid

from ...data_tools import DataCollector
from .progress_io import ProgressIO


class ProgressBoard:
    def __init__(self, filter_file=None, hide_progress_data=True):
        self.filter_file = filter_file
        self.hide_progress_data = hide_progress_data

        self.uuid = uuid.uuid4().hex
        self.search_ids = []

        self._io_ = ProgressIO(
            "./", hide_progress_data=hide_progress_data, verbosity=False
        )

    def init_paths(self, search_id, search_space):
        self._io_.remove_progress(search_id)
        data_c = DataCollector(self._io_.get_progress_data_path(search_id))

        self.search_ids.append(search_id)
        if self.filter_file:
            self._io_.create_filter(search_id, search_space)

        return data_c

    def open_dashboard(self):
        abspath = os.path.abspath(__file__)
        dir_ = os.path.dirname(abspath)

        paths = " ".join(self.search_ids)
        open_streamlit = "streamlit run " + dir_ + "/run_streamlit.py " + paths

        # from: https://stackoverflow.com/questions/7574841/open-a-terminal-from-python
        os.system('gnome-terminal -x bash -c " ' + open_streamlit + ' " ')
