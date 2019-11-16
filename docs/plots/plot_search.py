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
    {"HillClimbing": {"epsilon": 0.03}},
    {"HillClimbing": {"epsilon": 0.1}},
    {"StochasticHillClimbing": {"p_down": 0.5}},
    {"StochasticHillClimbing": {"p_down": 0.8}},
    {"TabuSearch": {"tabu_memory": 3}},
    {"TabuSearch": {"tabu_memory": 10}},
    "RandomSearch",
    {"RandomRestartHillClimbing": {"n_restarts": 10}},
    {"RandomRestartHillClimbing": {"n_restarts": 5}},
    {"RandomAnnealing": {"epsilon": 1}},
    {"RandomAnnealing": {"epsilon": 0.5}},
    {"RandomAnnealing": {"epsilon": 0.3}},
    {"RandomAnnealing": {"epsilon": 0.1}},
    {"SimulatedAnnealing": {"annealing_rate": 0.99}},
    {"SimulatedAnnealing": {"annealing_rate": 0.9}},
    {"StochasticTunneling": {"gamma": 0.1}},
    {"StochasticTunneling": {"gamma": 3}},
    {"ParallelTempering": {"system_temperatures": [0.1, 0.5, 1, 3]}},
    {"ParallelTempering": {"system_temperatures": [0.05, 0.3, 0.5, 1, 3, 5, 9]}},
    {"ParallelTempering": {"system_temperatures": [0.01, 1, 100]}},
    {"ParticleSwarm": {"n_particles": 10}},
    {"ParticleSwarm": {"n_particles": 20}},
    {"EvolutionStrategy": {"individuals": 10}},
    {"EvolutionStrategy": {"individuals": 4}},
    {
        "EvolutionStrategy": {
            "individuals": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.9,
        }
    },
    {
        "EvolutionStrategy": {
            "individuals": 10,
            "mutation_rate": 0.9,
            "crossover_rate": 0.1,
        }
    },
    # "Bayesian",
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

    if isinstance(opt, dict):

        print(str(list(opt.keys())[0]))

        if str(list(opt.keys())[0]) == "ParallelTempering":
            systems = len(list(opt[list(opt.keys())[0]].values())[0])
            print("systems", systems)
            n_iter_temp = int(n_iter / systems)

        if str(list(opt.keys())[0]) == "ParticleSwarm":
            n_part = list(opt[list(opt.keys())[0]].values())[0]
            n_iter_temp = int(n_iter / n_part)

        if str(list(opt.keys())[0]) == "EvolutionStrategy":
            n_pop = list(opt[list(opt.keys())[0]].values())[0]
            n_iter_temp = int(n_iter / n_pop)

    else:
        print(opt)

    opt_ = Hyperactive(
        search_config,
        optimizer=opt,
        n_iter=n_iter_temp,
        get_search_path=True,
        **opt_dict_temp
    )
    opt_.search(X, y)

    pos_list = opt_.pos_list
    score_list = opt_.score_list

    pos_list = np.array(pos_list)
    score_list = np.array(score_list)

    plt.figure(figsize=(5.5, 4.7))
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

    if isinstance(opt, dict):
        opt_key = list(opt.keys())[0]
        opt_title = str(opt_key) + "\n" + str(list(opt[opt_key].items()))
        opt_file_name = str(opt_key) + " " + str(list(opt[opt_key].items()))
    else:
        opt_title = opt
        opt_file_name = opt

    plt.title(opt_title)
    plt.xlabel("n_neighbors")
    plt.ylabel("leaf_size")

    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("./search_paths/" + opt_file_name + ".svg")
