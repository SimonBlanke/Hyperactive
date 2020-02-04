# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time
import tqdm
import hyperactive

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from hyperactive import Hyperactive


version = str("_v" + hyperactive.__version__)

#################################################################################################

runs = 1
n_iter = 10
cv = 5

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


def collect_data(runs, X, y, sklearn_model, opt_list, search_config, n_iter):
    time_c = time.time()

    data_runs = []
    for run in range(runs):
        print("\nRun nr.", run, "\n")
        time_opt = []

        start = time.perf_counter()
        for i in tqdm.tqdm(range(n_iter)):
            scores = cross_val_score(
                sklearn_model, X, y, scoring="accuracy", n_jobs=1, cv=cv
            )
        time_ = time.perf_counter() - start

        time_opt.append(time_)
        # data["No Opt"]["0"] = time_

        for opt_str in opt_list:
            print("optimizer:", opt_str, type(opt_str))

            n_iter_temp = n_iter
            if opt_str == "ParallelTempering":
                n_iter_temp = int(n_iter / 4)

            if opt_str == "ParticleSwarm":
                n_iter_temp = int(n_iter / 10)

            if opt_str == "EvolutionStrategy":
                n_iter_temp = int(n_iter / 10)

            opt = Hyperactive(X, y, memory=False)

            start = time.perf_counter()
            opt.search(search_config, n_iter=n_iter_temp, optimizer=opt_str)
            time_ = time.perf_counter() - start

            time_opt.append(time_)

        time_opt = np.array(time_opt)
        time_opt = time_opt / n_iter
        # time_opt = np.expand_dims(time_opt_norm, axis=0)

        data_runs.append(time_opt)

    data_runs = np.array(data_runs)
    print("\nCreate Dataframe\n")

    print("data_runs", data_runs, data_runs.shape)

    column_names = ["No Opt."] + opt_list
    data = pd.DataFrame(data_runs, columns=column_names)

    model_name = list(search_config.keys())[0]

    calc_optimizer_time_name = (
        "optimizer_calc_time_" + str(sklearn_model.__class__.__name__) + ".csv"
    )

    file_name = str(calc_optimizer_time_name)
    data.to_csv(file_name, index=False)

    print("data collecting time:", time.time() - time_c)


#################################################################################################
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris_data = load_iris()
iris_X, iris_y = iris_data.data, iris_data.target


def model(para, X, y):
    dtc = DecisionTreeClassifier(
        max_depth=para["max_depth"],
        min_samples_split=para["min_samples_split"],
        min_samples_leaf=para["min_samples_leaf"],
    )
    scores = cross_val_score(dtc, X, y, cv=cv)

    return scores.mean()


search_config_DTC = {
    model: {
        "max_depth": range(1, 21),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(1, 21),
    }
}

data_runs_dict_KNN = {
    "runs": runs,
    "X": iris_X,
    "y": iris_y,
    "sklearn_model": DecisionTreeClassifier(),
    "opt_list": opt_list,
    "search_config": search_config_DTC,
    "n_iter": n_iter,
}

collect_data(**data_runs_dict_KNN)

#################################################################################################
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
cancer_X, cancer_y = data.data, data.target


def model(para, X, y):
    gbc = GradientBoostingClassifier(n_estimators=para["n_estimators"])
    scores = cross_val_score(gbc, X, y, cv=cv)

    return scores.mean()


search_config_gbc = {model: {"n_estimators": range(99, 102)}}
data_runs_dict_gbc = {
    "runs": runs,
    "X": cancer_X,
    "y": cancer_y,
    "sklearn_model": GradientBoostingClassifier(n_estimators=100),
    "opt_list": opt_list,
    "search_config": search_config_gbc,
    "n_iter": n_iter,
}

# collect_data(**data_runs_dict_gbc)
