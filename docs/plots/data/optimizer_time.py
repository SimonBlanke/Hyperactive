# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import tqdm
import time
from hyperactive import Hyperactive

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


#################################################################################################

runs = 3
n_iter = 100

opt_list = [
    "HillClimbing",
    "StochasticHillClimbing",
    # "TabuSearch",
    "RandomSearch",
    "RandomRestartHillClimbing",
    "RandomAnnealing",
    "SimulatedAnnealing",
    "StochasticTunneling",
    "ParallelTempering",
    "ParticleSwarm",
    "EvolutionStrategy",
    # "Bayesian",
]

#################################################################################################


def collect_data(runs, X, y, opt_list, search_config, n_iter):
    time_c = time.time()

    data_runs_1 = []
    data_runs_2 = []
    for run in tqdm.tqdm(range(runs)):
        print("\nRun nr.", run, "\n")
        eval_time_list = []
        opt_time_list = []

        for key in opt_list:
            print("optimizer:", key)

            n_iter_temp = n_iter

            if key == "ParallelTempering":
                n_iter_temp = int(n_iter / 4)

            if key == "ParticleSwarm":
                n_iter_temp = int(n_iter / 10)

            if key == "EvolutionStrategy":
                n_iter_temp = int(n_iter / 10)

            opt_obj = Hyperactive(X, y, memory=False)
            opt_obj.search(search_config, optimizer=key, n_iter=n_iter_temp)

            model = list(search_config.keys())[0]

            eval_times = opt_obj.eval_times[model]
            opt_times = opt_obj.opt_times[model]

            eval_time = np.array(eval_times).sum()
            opt_time = np.array(opt_times).sum()

            eval_time_list.append(eval_time)
            opt_time_list.append(opt_time)

        eval_time_list = np.array(eval_time_list)
        opt_time_list = np.array(opt_time_list)

        data_runs_1.append(eval_time_list)
        data_runs_2.append(opt_time_list)

    data_runs_1 = np.array(data_runs_1)
    data_runs_2 = np.array(data_runs_2)

    print("\nCreate Dataframe\n")

    print("data_runs_1", data_runs_1, data_runs_1.shape)

    data = pd.DataFrame(data_runs_1, columns=opt_list)

    model_name = list(search_config.keys())[0]

    calc_optimizer_time_name = "eval_time_" + model_name.__name__

    file_name = str(calc_optimizer_time_name)
    data.to_csv(file_name, index=False)

    data = pd.DataFrame(data_runs_2, columns=opt_list)

    calc_optimizer_time_name = "opt_time_" + model_name.__name__

    file_name = str(calc_optimizer_time_name)
    data.to_csv(file_name, index=False)

    print("data collecting time:", time.time() - time_c)


#################################################################################################
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

iris_data = load_iris()
iris_X, iris_y = iris_data.data, iris_data.target


def model(para, X, y):
    dtc = GradientBoostingClassifier(max_depth=para["max_depth"])
    scores = cross_val_score(dtc, X, y, cv=2)

    return scores.mean()


search_config = {model: {"max_depth": range(2, 20)}}


data_runs_dict_KNN = {
    "runs": runs,
    "X": iris_X,
    "y": iris_y,
    "opt_list": opt_list,
    "search_config": search_config,
    "n_iter": n_iter,
}

data_runs = collect_data(**data_runs_dict_KNN)
