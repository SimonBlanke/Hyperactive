# sns.set(color_codes=True)
# sns.set_palette(sns.color_palette("RdBu", n_colors=7))
# sns.set(rc={'figure.figsize':(12, 9)})

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from hyperactive import HillClimbingOptimizer
from hyperactive import StochasticHillClimbingOptimizer
from hyperactive import TabuOptimizer
from hyperactive import RandomSearchOptimizer
from hyperactive import RandomRestartHillClimbingOptimizer
from hyperactive import RandomAnnealingOptimizer
from hyperactive import SimulatedAnnealingOptimizer
from hyperactive import StochasticTunnelingOptimizer
from hyperactive import ParallelTemperingOptimizer
from hyperactive import ParticleSwarmOptimizer
from hyperactive import EvolutionStrategyOptimizer
from hyperactive import BayesianOptimizer

breast_cancer_data = load_breast_cancer()
X = breast_cancer_data.data
y = breast_cancer_data.target

opt_list = {
    "Hill Climbing": HillClimbingOptimizer,
    "Stoch. Hill Climbing": StochasticHillClimbingOptimizer,
    "Tabu Search": TabuOptimizer,
    "Random Search": RandomSearchOptimizer,
    "Rand. Rest. Hill Climbing": RandomRestartHillClimbingOptimizer,
    "Random Annealing": RandomAnnealingOptimizer,
    "Simulated Annealing": SimulatedAnnealingOptimizer,
    "Stochastic Tunneling": StochasticTunnelingOptimizer,
    "Parallel Tempering": ParallelTemperingOptimizer,
    "Particle Swarm": ParticleSwarmOptimizer,
    "Evolution Strategy": EvolutionStrategyOptimizer,
    "Bayesian Optimization": BayesianOptimizer,
}

search_config = {
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(1, 102, 1),
        "max_depth": range(1, 32, 1),
    }
}
n_iter = 150

opt_dict = {"cv": 5, "n_jobs": 1, "memory": False, "verbosity": 0}


def _plot(plt, pos, score):
    df = pd.DataFrame(
        {"n_estimators": pos[:, 0], "max_depth": pos[:, 1], "score": score}
    )

    # plot
    plt.plot(
        "n_estimators",
        "max_depth",
        data=df,
        linestyle="-",
        marker=",",
        color="gray",
        alpha=0.33,
    )
    plt.scatter(
        df["n_estimators"],
        df["max_depth"],
        c=df["score"],
        marker="H",
        s=50,
        vmin=0.88,
        vmax=0.98,
    )

    return plt


for opt in opt_list:
    n_iter_temp = n_iter
    opt_dict_temp = opt_dict

    if opt == "Parallel Tempering":
        n_iter_temp = int(n_iter / 10)
        opt_dict_temp["system_temps"] = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

    if opt == "Particle Swarm":
        n_iter_temp = int(n_iter / 10)
        opt_dict_temp["n_part"] = 10

    if opt == "Evolution Strategy":
        n_iter_temp = int(n_iter / 10)
        opt_dict_temp["individuals"] = 10

    opt_ = opt_list[opt]
    opt_ = opt_(
        search_config, n_iter=n_iter_temp, get_search_path=True, **opt_dict_temp
    )
    opt_.fit(X, y)

    pos_list = opt_.pos_list
    score_list = opt_.score_list

    pos_list = np.array(pos_list)
    score_list = np.array(score_list)

    plt.figure(figsize=(10, 4))
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
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")

    plt.xlim((0, 100))
    plt.ylim((0, 30))
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("search_path_" + opt + ".png")
