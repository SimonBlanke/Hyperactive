import numpy as np
from hyperactive import Hyperactive, BayesianOptimizer

from gradient_free_optimizers import RandomRestartHillClimbingOptimizer

def meta_opt(opt_para):
    scores = []

    for i in range(33):

        def ackley_function(para):
            x = para["x"]
            y = para["y"]
            loss1 = - 20 * np.exp(-0.2 * np.sqrt(0.5 * (x * x + y * y)))
            loss2 = -np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            loss3 = np.exp(1)
            loss4 = 20

            loss = loss1 + loss2 + loss3 + loss4

            return -loss

        dim_size = np.arange(-6, 6, 0.01)

        search_space = {
            "x": dim_size,
            "y": dim_size,
        }

        opt = RandomRestartHillClimbingOptimizer(
            search_space,
            epsilon=opt_para["epsilon"],
            n_neighbours=opt_para["n_neighbours"],
            n_iter_restart=opt_para["n_iter_restart"],
        )
        opt.search(
            ackley_function,
            n_iter=100,
            random_state=i,
            verbosity=False,
        )

        scores.append(opt.best_score)

    return np.array(scores).sum()


search_space = {
    "epsilon": list(np.arange(0.01, 0.1, 0.01)),
    "n_neighbours": list(range(1, 10)),
    "n_iter_restart": list(range(2, 12)),
}


optimizer = BayesianOptimizer()

hyper = Hyperactive()
hyper.add_search(meta_opt, search_space, n_iter=120, optimizer=optimizer)
hyper.run()
