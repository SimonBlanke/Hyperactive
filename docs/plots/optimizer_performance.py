import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from hyperactive import Hyperactive
from test_functions import (
    sphere_function_search_config,
    rastrigin_function_search_config,
)

search_config_list = [rastrigin_function_search_config]

X, y = np.array([0]), np.array([0])

optimizer_list = [
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
    # "Bayesian",
]


losses = []
for optimizer in tqdm.tqdm(optimizer_list):
    loss_opt = []

    for search_config in search_config_list:
        loss_avg = []

        for i in range(10):
            opt = Hyperactive(X, y, memory="short", random_state=i, verbosity=0)
            opt.search(search_config, n_iter=100, optimizer=optimizer)

            model = list(search_config.keys())[0]
            loss = opt.best_scores[model]

            loss_avg.append(loss)

        loss_avg = np.array(loss_avg).mean()
        loss_opt.append(loss_avg)

    loss_opt = np.array(loss_opt).sum()
    losses.append(loss_opt)

losses = np.array(losses).reshape(-1, 1)
"""
scaler = MinMaxScaler()
losses = scaler.fit_transform(losses)
"""
for loss, optimizer in zip(losses, optimizer_list):
    print("\n", optimizer)
    print("loss=", loss[0] * 100)
