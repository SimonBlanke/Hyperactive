# sns.set(color_codes=True)
# sns.set_palette(sns.color_palette("RdBu", n_colors=7))
# sns.set(rc={'figure.figsize':(12, 9)})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from hyperactive import Hyperactive

data = load_iris()
X = data.data
y = data.target

opt_list = [
    "HillClimbing",
    "StochasticHillClimbing",
    "TabuSearch",
    "RandomSearch",
    "RandomRestartHillClimbing",
    "RandomAnnealing",
    "SimulatedAnnealing",
    "StochasticTunneling",
    "ParallelTempering",
    "ParticleSwarm",
    "EvolutionStrategy",
    "Bayesian",
]


def model(para, X, y):
    model = KNeighborsClassifier(
        n_neighbors=para["n_neighbors"], leaf_size=para["leaf_size"]
    )
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {model: {"n_neighbors": range(1, 100), "leaf_size": range(1, 100)}}

n_iter = 100

opt_dict = {"memory": False, "verbosity": 0}


def _plot(plt, pos, score):
    df = pd.DataFrame(
        {"n_neighbors": pos[:, 0], "leaf_size": pos[:, 1], "score": score}
    )

    # plot
    plt.plot(
        "n_neighbors",
        "leaf_size",
        data=df,
        linestyle="-",
        marker=",",
        color="gray",
        alpha=0.33,
    )
    plt.scatter(
        df["n_neighbors"],
        df["leaf_size"],
        c=df["score"],
        marker="H",
        s=30,
        vmin=0.89,
        vmax=0.99,
    )

    return plt


for opt in opt_list:
    n_iter_temp = n_iter
    opt_dict_temp = opt_dict

    if opt == "ParallelTempering":
        n_iter_temp = int(n_iter / 10)

    if opt == "ParticleSwarm":
        n_iter_temp = int(n_iter / 10)

    if opt == "EvolutionStrategy":
        n_iter_temp = int(n_iter / 10)

    opt_ = Hyperactive(
        search_config,
        optimizer=opt,
        n_iter=n_iter_temp,
        get_search_path=True,
        # **opt_dict_temp
    )
    opt_.search(X, y)

    pos_list = opt_.pos_list
    score_list = opt_.score_list

    pos_list = np.array(pos_list)
    score_list = np.array(score_list)

    plt.figure(figsize=(15, 5))
    plt.set_cmap("jet")

    pos_list = np.swapaxes(pos_list, 0, 1)
    score_list = np.swapaxes(score_list, 0, 1)

    # print("\npos_list\n", pos_list, pos_list.shape)
    # print("score_list\n", score_list, score_list.shape)

    for pos, score in zip(pos_list, score_list):
        # print(pos[:, 0])
        # print(pos[:, 1])
        # print(score, "\n")
        plt = _plot(plt, pos, score)

    plt.title(opt)
    plt.xlabel("n_neighbors")
    plt.ylabel("leaf_size")

    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("search_path_" + opt + ".svg")
