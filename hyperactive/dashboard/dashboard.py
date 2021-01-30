# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os

# from ..ltm_data_path import ltm_data_path


class Dashboard:
    def __init__(self, path):
        self.path = path
        """
        if path is None:
            self.ltm_data_dir = ltm_data_path()
        else:
            self.ltm_data_dir = path + "/ltm_data/"
        """

    def open(
        self,
        plots=[
            "score_statistics",
            "1d_scatter",
            "2d_scatter",
            "3d_scatter",
            "parallel_coordinates",
        ],
    ):
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)

        streamlit_plot_args = " ".join(plots)

        command = (
            "streamlit run "
            + dname
            + "/st_script.py "
            + self.path
            + " "
            + streamlit_plot_args
        )
        os.system(command)
