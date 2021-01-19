# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import sys
import os
import dill
import glob
import inspect
import imageio

import hiplot as hip
import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def meta_data_path():
    current_path = os.path.realpath(__file__)
    return current_path.rsplit("/", 1)[0] + "/"


def _pkl_valid(pkl_path):
    return os.path.isfile(pkl_path) and os.path.getsize(pkl_path) > 0


def search_data_path(ltm_data_path, model_name, study_name):
    return ltm_data_path + model_name + ":" + study_name + "/search_data.pkl"


def objective_function_path(ltm_data_path, model_name, study_name):
    return ltm_data_path + model_name + ":" + study_name + "/objective_function.pkl"


def dill_load(path):
    if _pkl_valid(path):
        with open(path, "rb") as handle:
            object_ = dill.load(handle)

        return object_


def read_model_study_names(path):
    model_paths = glob.glob(path + "*/")

    model_names = []
    study_names = []

    study2model_name = {}

    for model_path in model_paths:
        model_study_name = model_path.rsplit("/", 2)[1]

        model_name = model_study_name.rsplit(":", 2)[0]
        study_name = model_study_name.rsplit(":", 2)[1]

        if study_name in study2model_name:
            study2model_name[study_name].append(model_name)
        else:
            study2model_name[study_name] = [model_name]

        model_names.append(model_name)
        study_names.append(study_name)

    return study2model_name


def get_model_string(function):
    return inspect.getsource(function)


st.set_page_config(page_title="Hyperactive Dashboard", layout="wide")
ltm_data_path = sys.argv[1]


st.title("Hyperactive Dashboard")
st.text("")
st.text("")

study2model_name = read_model_study_names(ltm_data_path)

col1, col2 = st.beta_columns(2)

col1.subheader("Study name")
study_name_select = col1.selectbox(
    "  ",
    list(study2model_name.keys()),
    index=0,
)
col1.subheader("Model name")
model_name_select = col1.selectbox(
    " ",
    study2model_name[study_name_select],
    # format_func=selectbox_model_names.get,
    index=0,
)

model_study_path_ = search_data_path(
    ltm_data_path, model_name_select, study_name_select
)
objective_function_path_ = objective_function_path(
    ltm_data_path, model_name_select, study_name_select
)

search_data = dill_load(model_study_path_)
objective_function_ = dill_load(objective_function_path_)
model_string = get_model_string(objective_function_)


if len(search_data) > 0:
    para_names = search_data.columns.drop("score")

    col2.subheader("Objective function:")
    col2.code(model_string)

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

    # ----------------- # Score statistics
    col1, col2 = st.beta_columns(2)

    col1.header("Score statistics")
    col1.text("")
    col2.text("")

    col1.table(df_data)

    fig = px.histogram(
        search_data, x="score", nbins=int(len(search_data))
    ).update_layout(width=1000, height=300)
    col2.plotly_chart(fig)

    # ----------------- # N-dimensional Scatter plots
    col1, col2 = st.beta_columns(2)

    col1.header("N-dimensional Scatter plots")
    col1.text("")
    col2.text("")

    # ----------------- # 1D scatter plot
    col1, col2 = st.beta_columns(2)

    scatter1_para1 = col1.selectbox(
        "1D scatter plot parameter 1",
        para_names,
    )

    fig = px.scatter(
        search_data, x=scatter1_para1, y=search_data["score"]
    ).update_layout(width=1000, height=600)
    col2.plotly_chart(fig)

    # ----------------- # 2D scatter plot
    col1, col2 = st.beta_columns(2)

    scatter2_para1 = col1.selectbox(
        "2D scatter plot parameter 1",
        para_names,
    )
    scatter2_para2 = col1.selectbox(
        "2D scatter plot parameter 2",
        para_names,
    )

    fig = px.scatter(
        search_data, x=scatter2_para1, y=scatter2_para2, color="score"
    ).update_layout(width=1000, height=600)
    col2.plotly_chart(fig)

    # ----------------- # 3D scatter plot
    col1, col2 = st.beta_columns(2)

    scatter3_para1 = col1.selectbox(
        "3D scatter plot parameter 1",
        para_names,
    )
    scatter3_para2 = col1.selectbox(
        "3D scatter plot parameter 2",
        para_names,
    )
    scatter3_para3 = col1.selectbox(
        "3D scatter plot parameter 3",
        para_names,
    )

    fig = px.scatter_3d(
        search_data,
        x=scatter3_para1,
        y=scatter3_para2,
        z=scatter3_para3,
        color="score",
    ).update_layout(width=1000, height=600)
    col2.plotly_chart(fig)

    # ----------------- # Parallel Corrdinates
    col1, col2 = st.beta_columns(2)

    col1.header("Parallel Corrdinates")
    col1.text("")
    col2.text("")

    xp = hip.Experiment.from_dataframe(search_data)
    ret_val = xp.display_st(key="key1")

else:
    st.text("No search data found")