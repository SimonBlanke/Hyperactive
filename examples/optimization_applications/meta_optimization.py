import numpy as np
from sklearn.datasets import load_breast_cancer
from hyperactive import Hyperactive

from gradient_free_optimizers import EvolutionStrategyOptimizer

data = load_breast_cancer()
X, y = data.data, data.target


def meta_opt(opt_para):
    scores = []

    for i in range(25):

        def sphere_function(para):
            loss = []
            for key in para.keys():
                loss.append(para[key] * para[key])

            return -np.array(loss).sum()

        dim_size = np.arange(-10, 10, 0.01)

        search_space = {
            "x1": dim_size,
            "x2": dim_size,
        }

        opt = EvolutionStrategyOptimizer(
            search_space,
            mutation_rate=opt_para["mutation_rate"],
            crossover_rate=opt_para["crossover_rate"],
            initialize={"random": opt_para["individuals"]},
        )
        opt.search(
            sphere_function,
            n_iter=100,
            random_state=i,
            verbosity=False,
        )

        scores.append(opt.best_score)

    return np.array(scores).sum()


search_space = {
    "individuals": list(range(2, 11)),
    "mutation_rate": list(np.arange(0, 1, 0.1)),
    "crossover_rate": list(np.arange(0, 1, 0.1)),
}


hyper = Hyperactive()
hyper.add_search(meta_opt, search_space, n_iter=50)
hyper.run()
