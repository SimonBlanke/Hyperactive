# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt


color_scale = px.colors.sequential.Jet


def parallel_coordinates_plotly(*args, plotly_width=1200, plotly_height=540, **kwargs):
    fig = px.parallel_coordinates(*args, **kwargs, color_continuous_scale=color_scale)
    fig.update_layout(autosize=False, width=plotly_width, height=plotly_height)

    return fig


def filter_data(filter, df, columns):
    if len(df) > 1:
        for column in columns:
            if column not in list(filter["parameter"]):
                continue

            filter_ = filter[filter["parameter"] == column]
            lower, upper = (
                filter_["lower bound"].values[0],
                filter_["upper bound"].values[0],
            )

            col_data = df[column]

            if lower == "lower":
                lower = np.min(col_data)
            else:
                lower = float(lower)

            if upper == "upper":
                upper = np.max(col_data)
            else:
                upper = float(upper)

            df = df[(df[column] >= lower) & (df[column] <= upper)]

    return df


def main():
    try:
        st.set_page_config(page_title="Hyperactive Progress Board", layout="wide")
    except:
        pass

    search_ids = sys.argv[1:]

    search_id_dict = {}
    for search_id in search_ids:
        search_id_dict[search_id] = {}

        progress_data_path = "./progress_data_" + search_id + ".csv~"
        filter_path = "./filter_" + search_id + ".csv"

        if os.path.isfile(progress_data_path):
            search_id_dict[search_id]["progress_data"] = pd.read_csv(progress_data_path)
        if os.path.isfile(filter_path):
            search_id_dict[search_id]["filter"] = pd.read_csv(filter_path)

    for search_id in search_id_dict.keys():
        progress_data = search_id_dict[search_id]["progress_data"]
        filter = search_id_dict[search_id]["filter"]

        st.title(search_id)
        st.components.v1.html(
            """<hr style="height:1px;border:none;color:#333;background-color:#333;" /> """,
            height=10,
        )

        col1, col2 = st.beta_columns([1, 2])

        progress_data_f = progress_data[
            ~progress_data.isin([np.nan, np.inf, -np.inf]).any(1)
        ]

        nth_iter = progress_data_f["nth_iter"]
        score_best = progress_data_f["score_best"]
        nth_process = list(progress_data_f["nth_process"])

        if np.all(nth_process == nth_process[0]):
            fig, ax = plt.subplots()
            plt.plot(nth_iter, score_best)
            col1.pyplot(fig)
        else:
            fig, ax = plt.subplots()
            ax.set_xlabel("nth iteration")
            ax.set_ylabel("score")

            for i in np.unique(nth_process):
                nth_iter_p = nth_iter[nth_process == i]
                score_best_p = score_best[nth_process == i]
                plt.plot(nth_iter_p, score_best_p, label=str(i) + ". process")
            plt.legend()
            col1.pyplot(fig)

        progress_data_f.drop(
            ["nth_iter", "score_best", "nth_process"], axis=1, inplace=True
        )
        prog_data_columns = list(progress_data_f.columns)

        progress_data_f = filter_data(filter, progress_data_f, prog_data_columns)

        # remove score
        prog_data_columns.remove("score")

        fig = parallel_coordinates_plotly(
            progress_data_f, dimensions=prog_data_columns, color="score"
        )
        col2.plotly_chart(fig)

        for _ in range(3):
            st.write(" ")

    time.sleep(1)
    st.experimental_rerun()


if __name__ == "__main__":
    main()
