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
    file_name_1 = "./data/optimizer_calc_time_GradientBoostingClassifier.csv"

    iter_time_model = pd.read_csv(file_name_1, header=0)

    columns = iter_time_model.columns[1:]
    eval_time = iter_time_model.values[:, 0]
    iter_time = iter_time_model.values[:, 1:]

    print("eval_time", eval_time)
    print("iter_time", iter_time, iter_time.shape)

    eval_time_mean = eval_time.mean(axis=0)
    eval_time_std = eval_time.std(axis=0)

    eval_times = np.ones(eval_time_mean.shape)

    opt_time_mean = (iter_time - eval_time_mean).mean(axis=0) / eval_time_mean
    opt_time_std = iter_time.std(axis=0)

    ind = np.arange(opt_time_mean.shape[0])  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    plt.figure(figsize=(15, 5))

    print("\neval_times\n", eval_times)
    print("\nopt_time_mean\n", opt_time_mean)

    p2 = plt.bar(ind, opt_time_mean, width, bottom=eval_times, yerr=opt_time_std)
    p1 = plt.bar(ind, eval_times, width, yerr=eval_time_std)

    plt.ylabel("Time")
    plt.title(title)
    plt.xticks(ind, columns, rotation=75)
    # plt.yticks()
    plt.legend((p1[0], p2[0]), ("Eval time", "Opt time"))

    plt.tight_layout()
    plt.savefig("performance.png", dpi=400)

    """
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
    """


plot_optimizer_time("model", 0, 2, 0.25, "title")
