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
n_iter = 5

opt_dict = {"n_jobs": 1, "memory": False, "verbosity": 0}

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

#################################################################################################


def collect_data(runs, X, y, opt_list, search_config, n_iter, opt_dict):
    time_c = time.time()

    data_runs_1 = []
    data_runs_2 = []
    for run in tqdm.tqdm(range(runs)):
        print("\nRun nr.", run, "\n")
        total_time_list = []
        eval_time_list = []

        for key in opt_list:
            print("optimizer:", key)

            n_iter_temp = n_iter
            opt_dict_temp = opt_dict

            if key == "ParallelTempering":
                n_iter_temp = int(n_iter / 10)

            if key == "ParticleSwarm":
                n_iter_temp = int(n_iter / 10)

            if key == "EvolutionStrategy":
                n_iter_temp = int(n_iter / 10)

            opt_obj = Hyperactive(
                search_config, optimizer=key, n_iter=n_iter_temp, **opt_dict_temp
            )
            opt_obj.search(X, y)
            total_time = opt_obj.get_total_time()
            eval_time = opt_obj.get_eval_time()

            total_time_list.append(total_time)
            eval_time_list.append(eval_time)

        total_time_list = np.array(total_time_list)
        eval_time_list = np.array(eval_time_list)

        data_runs_1.append(total_time_list)
        data_runs_2.append(eval_time_list)

    data_runs_1 = np.array(data_runs_1)
    data_runs_2 = np.array(data_runs_2)

    print("\nCreate Dataframe\n")

    print("data_runs_1", data_runs_1, data_runs_1.shape)

    data = pd.DataFrame(data_runs_1, columns=opt_list)

    model_name = list(search_config.keys())[0]

    calc_optimizer_time_name = "total_time_" + model_name.__name__

    file_name = str(calc_optimizer_time_name)
    data.to_csv(file_name, index=False)

    data = pd.DataFrame(data_runs_2, columns=opt_list)

    calc_optimizer_time_name = "eval_time_" + model_name.__name__

    file_name = str(calc_optimizer_time_name)
    data.to_csv(file_name, index=False)

    print("data collecting time:", time.time() - time_c)


#################################################################################################
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier

iris_data = load_iris()
iris_X, iris_y = iris_data.data, iris_data.target


def model(para, X, y):
    model = GradientBoostingClassifier(n_estimators=para["n_estimators"])
    scores = cross_val_score(model, X, y, cv=3)

    return scores.mean()


search_config = {model: {"n_estimators": range(2, 20)}}


data_runs_dict_KNN = {
    "runs": runs,
    "X": iris_X,
    "y": iris_y,
    "opt_list": opt_list,
    "search_config": search_config,
    "n_iter": n_iter,
    "opt_dict": opt_dict,
}

data_runs = collect_data(**data_runs_dict_KNN)
