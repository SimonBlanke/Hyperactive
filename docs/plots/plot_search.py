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
    {"HillClimbing": {"climb_dist": np.random.laplace}},
    {"HillClimbing": {"epsilon": 0.03}},
    {"HillClimbing": {"epsilon": 0.1}},
    {"StochasticHillClimbing": {"p_down": 0.5}},
    {"StochasticHillClimbing": {"p_down": 0.8}},
    {"TabuSearch": {"tabu_memory": 3}},
    {"TabuSearch": {"tabu_memory": 10}},
    "RandomSearch",
    {"RandomRestartHillClimbing": {"n_restarts": 10}},
    {"RandomRestartHillClimbing": {"n_restarts": 5}},
    {"RandomAnnealing": {"epsilon_mod": 10}},
    {"RandomAnnealing": {"epsilon_mod": 33}},
    {"RandomAnnealing": {"epsilon_mod": 100}},
    {"RandomAnnealing": {"epsilon_mod": 100, "annealing_rate": 0.9}},
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
    "Bayesian",
]

opt_para_names = [
    {"HillClimbing": {"climb_dist": "laplace"}},
    {"HillClimbing": {"epsilon": 0.03}},
    {"HillClimbing": {"epsilon": 0.1}},
    {"StochasticHillClimbing": {"p_down": 0.5}},
    {"StochasticHillClimbing": {"p_down": 0.8}},
    {"TabuSearch": {"tabu_memory": 3}},
    {"TabuSearch": {"tabu_memory": 10}},
    "RandomSearch",
    {"RandomRestartHillClimbing": {"n_restarts": 10}},
    {"RandomRestartHillClimbing": {"n_restarts": 5}},
    {"RandomAnnealing": {"epsilon_mod": 10}},
    {"RandomAnnealing": {"epsilon_mod": 33}},
    {"RandomAnnealing": {"epsilon_mod": 100}},
    {"RandomAnnealing": {"epsilon_mod": 100, "annealing_rate": 0.9}},
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
    "Bayesian",
]

if len(opt_list) != len(opt_para_names):
    print("\n--------->   Warning! List lengths do not match!\n")


def test_func(para, X, y):
    x1 = para["x"] - 50
    x2 = para["y"] - 50

    a = 20
    b = 0.2
    c = 2 * np.pi

    x1 = x1 / 10
    x2 = x2 / 10

    sum1 = x1 ** 2 + x2 ** 2
    sum2 = np.cos(c * x1) + np.cos(c * x2)

    term1 = -a * np.exp(-b * ((1 / 2.0) * sum1 ** (0.5)))
    term2 = -np.exp((1 / 2.0) * sum2)

    return 1 - (term1 + term2 + a + np.exp(1)) / 10


x_range = range(0, 100)

search_config = {test_func: {"x": x_range, "y": x_range}}

n_iter = 100


def _plot(plt, pos, score):
    df = pd.DataFrame({"X": pos[:, 0], "Y": pos[:, 1], "score": score})

    # plot
    plt.plot("X", "Y", data=df, linestyle="-", marker=",", color="gray", alpha=0.33)
    plt.scatter(df["X"], df["Y"], c=df["score"], marker="H", s=30, vmin=0, vmax=1)

    return plt


for opt, opt_para in zip(opt_list, opt_para_names):
    n_iter_temp = n_iter

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

    Xy = np.array([0])

    opt_ = Hyperactive(Xy, Xy, verbosity=10, memory=False, random_state=5)
    opt_.search(search_config, optimizer=opt, n_iter=n_iter_temp)

    pos_list = opt_.pos_list
    score_list = opt_.score_list

    pos_list = np.array(pos_list)
    score_list = np.array(score_list)

    plt.figure(figsize=(5.5, 4.7))
    plt.set_cmap("jet")

    pos_list = np.swapaxes(pos_list, 0, 1)
    score_list = np.swapaxes(score_list, 0, 1)

    for pos, score in zip(pos_list, score_list):
        plt = _plot(plt, pos, score)

    if isinstance(opt, dict):
        opt_key = list(opt.keys())[0]
        opt_title = r'$\bf{' + str(opt_key) + '}$'

        for key in opt[opt_key].keys():
            opt_title = opt_title + "\n" + key + ": " + str(opt_para[opt_key][key])

        opt_file_name = str(opt_key) + " " + str(list(opt_para[opt_key].items()))
    else:
        opt_title = opt
        opt_file_name = opt

    plt.title(opt_title)
    plt.xlabel("X")
    plt.ylabel("Y")

    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("./search_paths/temp/" + opt_file_name + ".png")
