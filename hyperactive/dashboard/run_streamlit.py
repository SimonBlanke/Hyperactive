# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt


from bokeh.plotting import figure

color_scale = px.colors.sequential.Jet


def parallel_coordinates_plotly(*args, plotly_width=1200, plotly_height=540, **kwargs):
    fig = px.parallel_coordinates(*args, **kwargs, color_continuous_scale=color_scale)
    fig.update_layout(autosize=False, width=plotly_width, height=plotly_height)

    return fig


def main():
    try:
        st.set_page_config(page_title="Hyperactive Progress Board", layout="wide")
    except:
        pass

    search_ids = sys.argv[1:]

    progress_data_list = []
    filter_list = []
    for search_id in search_ids:
        progress_data_path = "./progress_data_" + search_id + ".csv~"
        filter_path = "./filter_" + search_id + ".csv"

        progress_data_list.append(pd.read_csv(progress_data_path))
        filter_list.append(pd.read_csv(filter_path))

    for progress_data, filter, search_id in zip(
        progress_data_list, filter_list, search_ids
    ):
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
        print("\n filter \n", filter)

        # filter data
        prog_data_columns = list(progress_data_f.columns)
        print("\n progress_data_f \n", progress_data_f, type(progress_data_f))

        if len(progress_data_f) > 1:
            for prog_data_column in prog_data_columns:
                print("\n prog_data_column \n", prog_data_column)
                print("\n parameter \n", filter["parameter"])

                if prog_data_column not in list(filter["parameter"]):
                    continue
                filter_ = filter[filter["parameter"] == prog_data_column]
                lower, upper = (
                    filter_["lower bound"].values[0],
                    filter_["upper bound"].values[0],
                )
                print("\n filter_ \n", filter_)
                print("\n lower \n", lower, type(lower))
                print("\n upper \n", upper)

                col_data = progress_data_f[prog_data_column]

                if lower == "lower":
                    lower = np.min(col_data)
                else:
                    lower = float(lower)

                if upper == "upper":
                    upper = np.max(col_data)
                else:
                    upper = float(upper)

                print("\n lower \n", lower)
                print("\n upper \n", upper)

                progress_data_f = progress_data_f[
                    (progress_data_f[prog_data_column] >= lower)
                    & (progress_data_f[prog_data_column] <= upper)
                ]

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
