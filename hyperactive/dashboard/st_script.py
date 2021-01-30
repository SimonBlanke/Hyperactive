# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import sys
import glob

# import inspect
# import imageio

import hiplot as hip
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px


def _score_statistics(search_data):
    values_ = search_data["score"].values

    mean_ = values_.mean()
    std_ = values_.std()
    min_ = np.amin(values_)
    max_ = np.amax(values_)

    df_data = pd.DataFrame(
        [[mean_, std_, min_, max_]],
        index=["score"],
        columns=["mean", "std", "min", "max"],
    )

    col1, col2 = st.beta_columns(2)

    col1.header("Score statistics")
    col1.text("")
    col2.text("")

    col1.table(df_data)

    def _score_statistics_plot(search_data):
        fig = px.histogram(
            search_data, x="score", nbins=int(len(search_data))
        ).update_layout(width=1000, height=300)
        col2.plotly_chart(fig)

    _score_statistics_plot(search_data)


def _1d_scatter(search_data):
    para_names = search_data.columns.drop("score")

    st.text("")
    col1, col2 = st.beta_columns(2)

    col1.header("1D Scatter plot")
    col1.text("")

    scatter1_para1 = col1.selectbox(
        "1D scatter plot parameter 1",
        para_names,
        index=0,
    )

    def _1d_scatter_plot(search_data):
        fig = px.scatter(
            search_data, x=scatter1_para1, y=search_data["score"]
        ).update_layout(width=1000, height=600)
        col2.plotly_chart(fig)

    _1d_scatter_plot(search_data)


def _2d_scatter(search_data):
    para_names = search_data.columns.drop("score")

    st.text("")
    col1, col2 = st.beta_columns(2)

    col1.header("2D Scatter plot")
    col1.text("")

    scatter2_para1 = col1.selectbox(
        "2D scatter plot parameter 1",
        para_names,
        index=0,
    )
    scatter2_para2 = col1.selectbox(
        "2D scatter plot parameter 2",
        para_names,
        index=1,
    )

    def _2d_scatter_plot(search_data):
        fig = px.scatter(
            search_data, x=scatter2_para1, y=scatter2_para2, color="score"
        ).update_layout(width=1000, height=600)
        col2.plotly_chart(fig)

    _2d_scatter_plot(search_data)


def _3d_scatter(search_data):
    para_names = search_data.columns.drop("score")

    st.text("")
    col1, col2 = st.beta_columns(2)

    col1.header("3D Scatter plot")
    col1.text("")

    scatter3_para1 = col1.selectbox(
        "3D scatter plot parameter 1",
        para_names,
        index=0,
    )
    scatter3_para2 = col1.selectbox(
        "3D scatter plot parameter 2",
        para_names,
        index=1,
    )
    scatter3_para3 = col1.selectbox(
        "3D scatter plot parameter 3",
        para_names,
        index=2,
    )

    def _3d_scatter_plot(search_data):
        fig = px.scatter_3d(
            search_data,
            x=scatter3_para1,
            y=scatter3_para2,
            z=scatter3_para3,
            color="score",
        ).update_layout(width=1000, height=600)
        col2.plotly_chart(fig)

    _3d_scatter_plot(search_data)


def _parallel_coordinates(search_data):
    st.text("")
    col1, col2 = st.beta_columns(2)

    col1.header("Parallel Corrdinates")
    col1.text("")
    col2.text("")

    xp = hip.Experiment.from_dataframe(search_data)
    ret_val = xp.display_st(key="key1")


plots_dict = {
    "score_statistics": _score_statistics,
    "1d_scatter": _1d_scatter,
    "2d_scatter": _2d_scatter,
    "3d_scatter": _3d_scatter,
    "parallel_coordinates": _parallel_coordinates,
}


st.set_page_config(page_title="Hyperactive Dashboard", layout="wide")
path = sys.argv[1]
streamlit_plot_args = sys.argv[2:]

search_data = pd.read_csv(path)
# print("\n search_data \n", search_data)

st.title("Hyperactive Dashboard")
st.text("")
st.text("")


if len(search_data) > 0:
    # --- # create plots in order of "streamlit_plot_args"
    for streamlit_plot_arg in streamlit_plot_args:
        plots_dict[streamlit_plot_arg](search_data)

else:
    st.subheader("---> Error: Search data is empty!")