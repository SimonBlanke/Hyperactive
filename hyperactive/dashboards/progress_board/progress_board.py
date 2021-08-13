# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import os
import uuid

from ...data_tools import DataCollector
from .progress_io import ProgressIO


class ProgressBoard:
    def __init__(self, filter_file=None):
        self.uuid = uuid.uuid4().hex

        self.filter_file = filter_file

        self.progress_ids = []
        self.search_ids = []

        self._io_ = ProgressIO(verbosity=False)
        self.progress_collectors = {}

    def create_lock(self, progress_id):
        path = self._io_.get_lock_file_path(progress_id)
        if not os.path.exists(path):
            os.mknod(path)

    def init_paths(self, search_id, search_space):
        progress_id = search_id + ":" + self.uuid

        if progress_id in self.progress_collectors:
            return self.progress_collectors[progress_id]

        self.search_ids.append(search_id)
        self.progress_ids.append(progress_id)

        self._io_.remove_progress(progress_id)
        self.create_lock(progress_id)
        data_c = DataCollector(self._io_.get_progress_data_path(progress_id))
        self.progress_collectors[progress_id] = data_c

        if self.filter_file:
            self._io_.create_filter(progress_id, search_space)

        return data_c

    def open_dashboard(self):
        abspath = os.path.abspath(__file__)
        dir_ = os.path.dirname(abspath)

        paths = " ".join(self.progress_ids)
        open_streamlit = "streamlit run " + dir_ + "/run_streamlit.py " + paths

        # from: https://stackoverflow.com/questions/7574841/open-a-terminal-from-python
        os.system('gnome-terminal -x bash -c " ' + open_streamlit + ' " ')
