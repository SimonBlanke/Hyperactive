# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import time
import tqdm
import hyperactive

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score


from hyperactive import HillClimbingOptimizer
from hyperactive import StochasticHillClimbingOptimizer
from hyperactive import RandomSearchOptimizer
from hyperactive import RandomRestartHillClimbingOptimizer
from hyperactive import RandomAnnealingOptimizer
from hyperactive import SimulatedAnnealingOptimizer
from hyperactive import StochasticTunnelingOptimizer
from hyperactive import ParticleSwarmOptimizer
from hyperactive import EvolutionStrategyOptimizer

# from hyperactive import BayesianOptimizer

version = str("_v" + hyperactive.__version__)

#################################################################################################


cancer_data = load_breast_cancer()
iris_data = load_iris()

cancer_X = cancer_data.data
cancer_y = cancer_data.target

iris_X = iris_data.data
iris_y = iris_data.target


#################################################################################################


search_config = {
    "sklearn.ensemble.GradientBoostingClassifier": {
        "n_estimators": range(99, 102),
        "max_depth": range(3, 4),
    }
}
runs = 30
n_iter = 300
n_jobs = 1
cv = 3
n_population = 10
warm_start = False
memory = False
verbosity = 0


HillClimbingOptimizer_inst = HillClimbingOptimizer(
    search_config, n_iter, cv=cv, n_jobs=n_jobs, memory=memory, verbosity=verbosity
)
StochasticHillClimbingOptimizer_inst = StochasticHillClimbingOptimizer(
    search_config, n_iter, cv=cv, n_jobs=n_jobs, memory=memory, verbosity=verbosity
)
RandomSearchOptimizer_inst = RandomSearchOptimizer(
    search_config, n_iter, cv=cv, n_jobs=n_jobs, memory=memory, verbosity=verbosity
)
RandomRestartHillClimbingOptimizer_inst = RandomRestartHillClimbingOptimizer(
    search_config, n_iter, cv=cv, n_jobs=n_jobs, memory=memory, verbosity=verbosity
)
RandomAnnealingOptimizer_inst = RandomAnnealingOptimizer(
    search_config, n_iter, cv=cv, n_jobs=n_jobs, memory=memory, verbosity=verbosity
)
SimulatedAnnealingOptimizer_inst = SimulatedAnnealingOptimizer(
    search_config, n_iter, cv=cv, n_jobs=n_jobs, memory=memory, verbosity=verbosity
)
StochasticTunnelingOptimizer_inst = StochasticTunnelingOptimizer(
    search_config, n_iter, cv=cv, n_jobs=n_jobs, memory=memory, verbosity=verbosity
)
ParticleSwarmOptimizer_inst = ParticleSwarmOptimizer(
    search_config,
    n_iter / n_population,
    cv=cv,
    n_jobs=n_jobs,
    memory=memory,
    verbosity=verbosity,
    n_part=n_population,
)
EvolutionStrategyOptimizer_inst = EvolutionStrategyOptimizer(
    search_config,
    n_iter / n_population,
    cv=cv,
    n_jobs=n_jobs,
    memory=memory,
    verbosity=verbosity,
    individuals=n_population,
)
# BayesianOptimizer_inst = BayesianOptimizer()

opt_list = {
    "Hill Climbing": HillClimbingOptimizer_inst,
    "Stoch. Hill Climbing": StochasticHillClimbingOptimizer_inst,
    "Random Search": RandomSearchOptimizer_inst,
    "Rand. Rest. Hill Climbing": RandomRestartHillClimbingOptimizer_inst,
    "Random Annealing": RandomAnnealingOptimizer_inst,
    "Simulated Annealing": SimulatedAnnealingOptimizer_inst,
    "Stochastic Tunneling": StochasticTunnelingOptimizer_inst,
    "Particle Swarm": ParticleSwarmOptimizer_inst,
    "Evolution Strategy": EvolutionStrategyOptimizer_inst,
    # "Bayesian": BayesianOptimizer_inst,
}


#################################################################################################

time_c = time.time()
data_runs = []
for run in range(runs):
    print("\nRun nr.", run, "\n")
    time_opt = []

    gbc = GradientBoostingClassifier()
    start = time.perf_counter()
    for i in tqdm.tqdm(range(n_iter)):

        scores = cross_val_score(
            gbc, iris_X, iris_y, scoring="accuracy", n_jobs=n_jobs, cv=cv
        )
    time_ = time.perf_counter() - start

    time_opt.append(time_)
    time.sleep(1)
    # data["No Opt"]["0"] = time_

    for key in opt_list.keys():
        print("\n optimizer:", key)
        opt_obj = opt_list[key]

        start = time.perf_counter()
        opt_obj.fit(iris_X, iris_y)
        time_ = time.perf_counter() - start

        time_opt.append(time_)
        time.sleep(1)

    time_opt = np.array(time_opt)
    time_opt = time_opt / n_iter
    # time_opt = np.expand_dims(time_opt_norm, axis=0)

    data_runs.append(time_opt)

data_runs = np.array(data_runs)
#################################################################################################
print("\nCreate Dataframe\n")

print("data_runs", data_runs, data_runs.shape)

column_names = ["No Opt."] + list(opt_list.keys())
data = pd.DataFrame(data_runs, columns=column_names)

calc_optimizer_time_name = "optimizer_calc_time"


file_name = str(calc_optimizer_time_name + version)
data.to_csv(file_name, index=False)

print("data collecting time:", time.time() - time_c)
