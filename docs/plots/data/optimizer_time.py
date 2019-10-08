# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time
import tqdm
import hyperactive

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score


from keras.datasets import cifar10

from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


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

version = str("_v" + hyperactive.__version__)

#################################################################################################

runs = 10
n_iter = 100

opt_dict = {"cv": 3, "n_jobs": 1, "memory": False, "verbosity": 0}

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

#################################################################################################


def collect_data(runs, X, y, sklearn_model, opt_list, search_config, n_iter, opt_dict):
    time_c = time.time()

    data_runs = []
    for run in range(runs):
        print("\nRun nr.", run, "\n")
        time_opt = []

        start = time.perf_counter()
        for i in tqdm.tqdm(range(n_iter)):
            scores = cross_val_score(
                sklearn_model,
                X,
                y,
                scoring="accuracy",
                n_jobs=opt_dict["n_jobs"],
                cv=opt_dict["cv"],
            )
        time_ = time.perf_counter() - start

        time_opt.append(time_)
        # data["No Opt"]["0"] = time_

        for key in opt_list.keys():
            print("optimizer:", key)

            n_iter_temp = n_iter
            opt_dict_temp = opt_dict

            if key == "Parallel Tempering":
                n_iter_temp = int(n_iter / 5)
                opt_dict_temp["system_temperatures"] = [0.1, 0.2, 0.01, 0.2, 0.01]

            if key == "Particle Swarm":
                n_iter_temp = int(n_iter / 5)
                opt_dict_temp["n_particles"] = 5

            if key == "Evolution Strategy":
                n_iter_temp = int(n_iter / 5)
                opt_dict_temp["individuals"] = 5

            opt_obj = opt_list[key](search_config, n_iter_temp, **opt_dict_temp)

            start = time.perf_counter()
            opt_obj.fit(X, y)
            time_ = time.perf_counter() - start

            time_opt.append(time_)

        time_opt = np.array(time_opt)
        time_opt = time_opt / n_iter
        # time_opt = np.expand_dims(time_opt_norm, axis=0)

        data_runs.append(time_opt)

    data_runs = np.array(data_runs)
    print("\nCreate Dataframe\n")

    print("data_runs", data_runs, data_runs.shape)

    column_names = ["No Opt."] + list(opt_list.keys())
    data = pd.DataFrame(data_runs, columns=column_names)

    model_name = list(search_config.keys())[0]

    calc_optimizer_time_name = "optimizer_calc_time_" + model_name

    file_name = str(calc_optimizer_time_name)
    data.to_csv(file_name, index=False)

    print("data collecting time:", time.time() - time_c)


#################################################################################################
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris_data = load_iris()
iris_X, iris_y = iris_data.data, iris_data.target

KNN = KNeighborsClassifier()

search_config_KNN = {"sklearn.neighbors.KNeighborsClassifier": {"n_neighbors": [5]}}

data_runs_dict_KNN = {
    "runs": runs,
    "X": iris_X,
    "y": iris_y,
    "sklearn_model": KNN,
    "opt_list": opt_list,
    "search_config": search_config_KNN,
    "n_iter": n_iter,
    "opt_dict": opt_dict,
}

# data_runs = collect_data(**data_runs_dict_KNN)

#################################################################################################
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier

cancer_data = load_breast_cancer()
cancer_X, cancer_y = cancer_data.data, cancer_data.target

DTC = DecisionTreeClassifier()

search_config_DTC = {"sklearn.tree.DecisionTreeClassifier": {"min_samples_split": [2]}}

data_runs_dict_DTC = {
    "runs": runs,
    "X": cancer_X,
    "y": cancer_y,
    "sklearn_model": DTC,
    "opt_list": opt_list,
    "search_config": search_config_DTC,
    "n_iter": n_iter,
    "opt_dict": opt_dict,
}

# data_runs = collect_data(**data_runs_dict_DTC)

#################################################################################################
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier

cancer_data = load_breast_cancer()
cancer_X, cancer_y = cancer_data.data, cancer_data.target

GBC = GradientBoostingClassifier()

search_config_GBC = {
    "sklearn.ensemble.GradientBoostingClassifier": {"n_estimators": [100]}
}

data_runs_dict_GBC = {
    "runs": runs,
    "X": cancer_X,
    "y": cancer_y,
    "sklearn_model": GBC,
    "opt_list": opt_list,
    "search_config": search_config_GBC,
    "n_iter": n_iter,
    "opt_dict": opt_dict,
}

# data_runs = collect_data(**data_runs_dict_GBC)

#################################################################################################
from sklearn.datasets import load_breast_cancer
from lightgbm import LGBMClassifier

cancer_data = load_breast_cancer()
cancer_X, cancer_y = cancer_data.data, cancer_data.target

LGBMC = LGBMClassifier(n_jobs=1)

search_config_LGBMC = {"lightgbm.LGBMClassifier": {"num_leaves": [31], "n_jobs": [1]}}

data_runs_dict_LGBMC = {
    "runs": runs,
    "X": cancer_X,
    "y": cancer_y,
    "sklearn_model": LGBMC,
    "opt_list": opt_list,
    "search_config": search_config_LGBMC,
    "n_iter": n_iter,
    "opt_dict": opt_dict,
}

data_runs = collect_data(**data_runs_dict_LGBMC)

#################################################################################################


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def make_model():
    from keras.applications import mobilenet_v2
    from keras.models import Model

    mobilenet_v2 = mobilenet_v2.MobileNetV2(weights="imagenet")
    mobilenet_v2_model = Model(inputs=mobilenet_v2.input, outputs=mobilenet_v2.output)
    mobilenet_v2_model.compile(
        loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]
    )

    return mobilenet_v2_model


mobilenet_v2_model = KerasClassifier(build_fn=make_model, batch_size=500, epochs=3)

search_config_mobilenet_v2 = {
    "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": ["adam"]},
    "keras.fit.0": {"epochs": [3], "batch_size": [500], "verbose": [0]},
    "keras.applications.mobilenet_v2.1": {"weights": ["imagenet"]},
}

data_runs_dict_mobilenet_v2 = {
    "runs": runs,
    "X": X_train,
    "y": y_train,
    "sklearn_model": mobilenet_v2_model,
    "opt_list": opt_list,
    "search_config": search_config_mobilenet_v2,
    "n_iter": n_iter,
    "opt_dict": opt_dict,
}

# data_runs = collect_data(**data_runs_dict_mobilenet_v2)
