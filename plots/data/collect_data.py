# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time

import pandas as pd

import hyperactive
from hyperactive import RandomSearch_Optimizer
from hyperactive import SimulatedAnnealing_Optimizer
from hyperactive import ParticleSwarm_Optimizer
from functools import partial
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

breast_cancer_data = load_breast_cancer()

X_train = breast_cancer_data.data
y_train = breast_cancer_data.target

optimizer_dict = {
    "Random": RandomSearch_Optimizer,
    "Sim": SimulatedAnnealing_Optimizer,
    "PSO": ParticleSwarm_Optimizer,
}

version = str("_v" + hyperactive.__version__)
n_jobs_range = range(1, 17)
n_iter = 10
n_iter_list = [n_iter]

n_estimator_search = 10
clf = RandomForestClassifier()
model_dict = {
    "sklearn.ensemble.RandomForestClassifier": {"n_estimators": n_estimator_search}
}
param_dist = {"n_estimators": n_estimator_search}

calc_optimizer_time_name = "optimizer_calc_time"


def calc_optimizer_time(optimizer_dict, n_iter_list, model_dict, name):

    index_names = n_iter_list
    column_names = ["Random (Sklearn)"] + list(optimizer_dict.keys())
    data = pd.DataFrame(index=index_names, columns=column_names)

    for n_iter in n_iter_list:
        for n in range(n_iter + 1):
            random_search = RandomizedSearchCV(
                clf, param_distributions=param_dist, n_iter=1, cv=5
            )
        function = partial(random_search.fit, X_train, y_train)
        cpu_time = get_cpu_time(function)

        data["Random (Sklearn)"][n_iter] = cpu_time

    for optimizer in optimizer_dict.keys():
        for n_iter in n_iter_list:

            opt = optimizer_dict[optimizer](model_dict, n_iter, "accuracy", n_jobs=1)
            function = partial(opt.fit, X_train, y_train)
            cpu_time = get_cpu_time(function)

            data[optimizer][n_iter] = cpu_time

    print(data)
    file_name = str(calc_optimizer_time_name + version)
    data.to_csv(file_name)


def get_cpu_time(function):
    current_time = time.perf_counter()
    function()
    return time.perf_counter() - current_time


calc_optimizer_time(optimizer_dict, n_iter_list, model_dict, calc_optimizer_time_name)
