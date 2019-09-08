# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# import matplotlib as mpl
import seaborn as sns

ml = MultipleLocator(5)
sns.set(style="whitegrid")


def plot_optimizer_time(model_name, y_min, y_max, step_major, title):
    file_name = "optimizer_calc_time_" + model_name

    data = pd.read_csv("./data/" + file_name, header=0)

    columns = data.columns
    values = data.values

    no_opt = values[:, 0]

    values_norm = values / no_opt[:, None]

    fig, ax = plt.subplots()
    plt.figure(figsize=(10, 8))
    # plt.grid(True)
    # plt.ylabel(r"$\dfrac{T_{norm}}{iteration}$", rotation=0)
    plt.ylabel(r"$T_{norm}$", rotation=0)

    data = pd.DataFrame(values_norm, columns=columns)
    ax = sns.barplot(data=data, alpha=0.75, capsize=0.1)
    # ax = sns.barplot(x=columns, y=values_norm, alpha=0.75, ci="sd")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75)
    ax.set_title(title)

    # ax.tick_params(axis="x", which="minor", bottom=False)
    # ax.yaxis.set_minor_locator(ml)

    # ax.set_yscale("log")
    y_min = y_min
    y_max = y_max
    step_major = step_major
    # step_minor = 0.025

    ax.set_ylim(y_min, y_max)
    ticks_major = list(np.arange(y_min, y_max, step_major))
    # ticks_minor = list(np.arange(y_min, y_max, step_minor))

    ax.yaxis.set_ticks(ticks_major)
    # ax.yaxis.set_ticks(ticks_minor, minor=True)

    ax.xaxis.set_label_coords(-0.05, 0.5)
    ax.get_yaxis().set_label_coords(-0.05, 1.1)

    # ax.yaxis.grid(which="minor", color="r", linestyle="-", linewidth=2)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig("optimizer_time_" + model_name + ".png")


plot_optimizer_time(
    "sklearn.neighbors.KNeighborsClassifier",
    y_min=0.50,
    y_max=4.50001,
    step_major=0.50,
    title="KNeighborsClassifier - iris_data",
)

plot_optimizer_time(
    "sklearn.ensemble.GradientBoostingClassifier",
    y_min=0.50,
    y_max=1.50001,
    step_major=0.25,
    title="GradientBoostingClassifier - cancer_data",
)

plot_optimizer_time(
    "sklearn.tree.DecisionTreeClassifier",
    y_min=0.50,
    y_max=3.000001,
    step_major=0.50,
    title="DecisionTreeClassifier - iris_data",
)

plot_optimizer_time(
    "lightgbm.LGBMClassifier",
    y_min=0.50,
    y_max=1.500001,
    step_major=0.25,
    title="LGBMClassifier - cancer_data",
)
