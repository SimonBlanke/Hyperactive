import time
import random
import datetime
import numpy as np
import pandas as pd

# import tensorflow as tf
# import keras

from functools import partial

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor

wine_data = load_wine()
iris_data = load_iris()

wine_X_train = wine_data.data
wine_y_train = wine_data.target

iris_X_train = iris_data.data
iris_y_train = iris_data.target

dataset_dict = {
    "wine": [wine_X_train, wine_y_train],
    "iris": [iris_X_train, iris_y_train],
}

search_dict_1 = {
    "sklearn.ensemble.RandomForestClassifier": {
        "n_estimators": [200],
        "criterion": ["gini", "entropy"],
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(2, 21),
        "bootstrap": [True, False],
    }
}


search_dict_2 = {
    "sklearn.tree.DecisionTreeClassifier": {
        #'criterion': ["gini", "entropy"],
        #'max_depth': range(1, 11),
        "min_samples_split": range(2, 21),
        "min_samples_leaf": range(2, 21),
    }
}


from hyperactive.hyperactive.random_search import RandomSearch_Optimizer
from hyperactive.hyperactive.simulated_annealing import SimulatedAnnealing_Optimizer
from hyperactive.hyperactive.particle_swarm_optimization import ParticleSwarm_Optimizer

optimizer_list = [
    RandomSearch_Optimizer,
    SimulatedAnnealing_Optimizer,
    ParticleSwarm_Optimizer,
]
n_jobs_range = range(1, 17)
n_searches_list = [10, 100]


def compare_random_search(n_jobs_range):
    n_searches = 100
    cpu_time_dict = {}

    for n_jobs in n_jobs_range:
        Optimizer = RandomSearch_Optimizer(
            search_dict_2, n_searches, "accuracy", n_jobs=n_jobs
        )

    return cpu_time_dict


def test_optimizers(optimizer_list, n_searches_list):
    opt_dict = {}

    for optimizer in optimizer_list:
        search_dict = {}

        for n_searches in n_searches_list:
            Optimizer = optimizer(search_dict_2, n_searches, "accuracy", n_jobs=1)
            function = partial(Optimizer.fit, iris_X_train, iris_y_train)
            cpu_time = get_cpu_time(function)

            search_dict[n_searches] = cpu_time

        opt_dict[optimizer] = search_dict

    return opt_dict


def test_multiprocessing(optimizer, n_jobs_range):
    n_searches = 100
    cpu_time_dict = {}

    for n_jobs in n_jobs_range:
        Optimizer = optimizer(search_dict_2, n_searches, "accuracy", n_jobs=n_jobs)
        function = partial(Optimizer.fit, iris_X_train, iris_y_train)

        cpu_time = get_cpu_time(function)

        cpu_time_dict[n_jobs] = cpu_time

    return cpu_time_dict


def get_cpu_time(function):
    current_time = time.process_time()
    function()
    return time.process_time() - current_time


# test_multiprocessing(SimulatedAnnealing_Optimizer, n_jobs_range)
test_optimizers(optimizer_list, n_searches_list)

"""
Optimizer = SimulatedAnnealing_Optimizer(search_dict_2, n_searches, 'accuracy', n_jobs=-1)
Optimizer = ParticleSwarm_Optimizer(search_dict_2, n_searches, 'accuracy', n_jobs=-1)

c_time = time.time()
Optimizer.fit(iris_X_train, iris_y_train)
print('Search time: ', round(time.time() - c_time, 3))

#prediction = Optimizer.predict(iris_X_train)


#scores = cross_val_score(RS_opt, iris_X_train, iris_y_train, scoring='accuracy', cv=5)
#print(scores.mean())
"""
