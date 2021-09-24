# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numbers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

try:
    from progress_io import ProgressIO
except:
    from .progress_io import ProgressIO


pd.options.mode.chained_assignment = "raise"


color_scale = px.colors.sequential.Jet


class StreamlitBackend:
    def __init__(self, progress_ids):
        self.progress_ids = progress_ids
        self.progress_id_dict = {}

        self._io_ = ProgressIO()

        for progress_id in progress_ids:
            self.progress_id_dict[progress_id] = {}

            self.progress_id_dict[progress_id]["prog_d"] = self._io_.load_progress(
                progress_id
            )
            self.progress_id_dict[progress_id]["filt_f"] = self._io_.load_filter(
                progress_id
            )

    def get_progress_data(self, progress_id):
        progress_data = self.progress_id_dict[progress_id]["prog_d"]

        if progress_data is None:
            return

        return progress_data[~progress_data.isin([np.nan, np.inf, -np.inf]).any(1)]

    def pyplot(self, progress_data):
        if progress_data is None or len(progress_data) <= 1:
            return None

        nth_iter = progress_data["nth_iter"]
        score_best = progress_data["score_best"]
        nth_process = list(progress_data["nth_process"])

        if np.all(nth_process == nth_process[0]):
            fig, ax = plt.subplots()
            plt.plot(nth_iter, score_best)
        else:
            fig, ax = plt.subplots()
            ax.set_xlabel("nth iteration")
            ax.set_ylabel("best score")

            for i in np.unique(nth_process):
                nth_iter_p = nth_iter[nth_process == i]
                score_best_p = score_best[nth_process == i]
                plt.plot(nth_iter_p, score_best_p, label=str(i) + ". process")
            plt.legend()

        return fig

    def filter_data(self, df, filter_df):
        prog_data_columns = list(df.columns)

        if len(df) > 1:
            for column in prog_data_columns:
                if column not in list(filter_df["parameter"]):
                    continue

                filter_ = filter_df[filter_df["parameter"] == column]
                lower, upper = (
                    filter_["lower bound"].values[0],
                    filter_["upper bound"].values[0],
                )

                col_data = df[column]

                if isinstance(lower, numbers.Number):
                    lower = float(lower)
                else:
                    lower = np.min(col_data)

                if isinstance(upper, numbers.Number):
                    upper = float(upper)
                else:
                    upper = np.max(col_data)

                df = df[(df[column] >= lower) & (df[column] <= upper)]

        return df

    def plotly(self, progress_data, progress_id):
        if progress_data is None or len(progress_data) <= 1:
            return None

        filter_df = self.progress_id_dict[progress_id]["filt_f"]

        progress_data = progress_data.drop(
            ["nth_iter", "score_best", "nth_process", "best"], axis=1
        )

        if filter_df is not None:
            progress_data = self.filter_data(progress_data, filter_df)

        # remove score
        prog_data_columns = list(progress_data.columns)
        prog_data_columns.remove("score")

        fig = px.parallel_coordinates(
            progress_data,
            dimensions=prog_data_columns,
            color="score",
            color_continuous_scale=color_scale,
        )

        return fig

    def table_plotly(self, search_data):
        df_len = len(search_data)

        headerColor = "#b5beff"
        rowEvenColor = "#e8e8e8"
        rowOddColor = "white"

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(search_data.columns),
                        fill_color=headerColor,
                        align="center",
                        font_size=18,
                        height=30,
                    ),
                    cells=dict(
                        values=[search_data[col] for col in search_data.columns],
                        # fill_color="lavender",
                        fill_color=[
                            [
                                rowOddColor,
                                rowEvenColor,
                            ]
                            * int((df_len / 2) + 1)
                        ],
                        align=["center"],
                        font_size=14,
                        height=30,
                    ),
                )
            ]
        )
        fig.update_layout(height=550)
        return fig

    def create_plots(self, progress_id):
        progress_data = self.get_progress_data(progress_id)

        pyplot_fig = self.pyplot(progress_data)
        plotly_fig = self.plotly(progress_data, progress_id)

        return pyplot_fig, plotly_fig

    def create_info(self, progress_id):
        progress_data = self.get_progress_data(progress_id)
        if progress_data is None or len(progress_data) <= 1:
            return None

        progress_data_best = progress_data.drop(
            ["nth_iter", "score_best", "nth_process", "best"], axis=1
        )

        progress_data_best = progress_data_best.sort_values("score")
        last_best = progress_data_best.tail(10)
        last_best = last_best.rename(
            columns={
                "score": "best 5 scores",
            }
        )

        return last_best
