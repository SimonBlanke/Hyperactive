# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import os
import uuid


class ProgressBoard:
    def __init__(self):
        self.uuid = uuid.uuid4().hex
        self.paths_list = []

    def open_dashboard(self):
        abspath = os.path.abspath(__file__)
        dir_ = os.path.dirname(abspath)

        paths = " ".join(self.paths_list)
        open_streamlit = "streamlit run " + dir_ + "/run_streamlit.py " + paths

        # from: https://stackoverflow.com/questions/7574841/open-a-terminal-from-python
        os.system("gnome-terminal -e 'bash -c \" " + open_streamlit + " \"'")
